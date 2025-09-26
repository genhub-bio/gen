use core::ops::Range;
use std::collections::{HashMap, HashSet};

use gen_core::{HashId, Strand, is_end_node, is_start_node};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    edge::{Edge, EdgeData},
    errors::OperationError,
    file_types::FileTypes,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    path_edge::PathEdge,
    sample::Sample,
    session_operations::{end_operation, start_operation},
    traits::Query,
};
use itertools::Itertools;
use rusqlite::Connection;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum GraphOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Invalid coordinate(s): {0}")]
    InvalidCoordinate(String),
    #[error("Region not found: {0}")]
    RegionNotFound(String),
    #[error("Path not found: {0}")]
    PathNotFound(String),
}

pub fn get_path(
    conn: &Connection,
    collection_name: &str,
    sample_name: Option<&str>,
    region_name: &str,
    backbone: Option<&str>,
) -> Result<Path, GraphOperationError> {
    let block_group_id = get_block_group_id(conn, collection_name, sample_name, region_name)?;

    if let Some(backbone) = backbone {
        let path = BlockGroup::get_path_by_name(conn, &block_group_id, backbone);
        if path.is_none() {
            return Err(GraphOperationError::PathNotFound(format!(
                "No path found with name {backbone}"
            )));
        }
        Ok(path.unwrap())
    } else {
        Ok(BlockGroup::get_current_path(conn, &block_group_id))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn derive_chunks(
    conn: &Connection,
    operation_conn: &Connection,
    collection_name: &str,
    parent_sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    backbone: Option<&str>,
    chunk_ranges: Vec<Range<i64>>,
) -> Result<Operation, GraphOperationError> {
    let mut session = start_operation(conn);
    let _new_sample = Sample::get_or_create(conn, new_sample_name);

    let parent_block_group_id =
        get_block_group_id(conn, collection_name, parent_sample_name, region_name)?;
    let current_path = get_path(
        conn,
        collection_name,
        parent_sample_name,
        region_name,
        backbone,
    )?;

    let current_path_length = current_path.length(conn);
    let current_intervaltree = current_path.intervaltree(conn);
    let current_path_edges = PathEdge::edges_for_path(conn, &current_path.id);

    let chunk_ranges_length = chunk_ranges.len();
    for (i, chunk_range) in chunk_ranges.clone().into_iter().enumerate() {
        let child_block_group_name = if chunk_ranges_length > 1 {
            format!("{}.{}", region_name, i + 1)
        } else {
            region_name.to_string()
        };

        let child_block_group = BlockGroup::create(
            conn,
            collection_name,
            Some(new_sample_name),
            child_block_group_name.as_str(),
        );
        let child_block_group_id = child_block_group.id;

        let start_coordinate = chunk_range.start;
        let end_coordinate = chunk_range.end;
        if (start_coordinate < 0 || start_coordinate > current_path_length)
            || (end_coordinate < 0 || end_coordinate > current_path_length)
        {
            return Err(GraphOperationError::InvalidCoordinate(format!(
                "Start and/or end coordinates ({start_coordinate}, {end_coordinate}) are out of range for the current path."
            )));
        }

        let mut blocks = current_intervaltree
            .query(Range {
                start: start_coordinate,
                end: end_coordinate,
            })
            .map(|x| x.value)
            .collect::<Vec<_>>();
        blocks.sort_by(|a, b| a.start.cmp(&b.start));
        let start_block = blocks[0];
        let start_node_coordinate =
            start_coordinate - start_block.start + start_block.sequence_start;
        let end_block = blocks[blocks.len() - 1];
        let end_node_coordinate = end_coordinate - end_block.start + end_block.sequence_start;

        let new_node_ids_by_old = BlockGroup::derive_subgraph(
            conn,
            collection_name,
            new_sample_name,
            &parent_block_group_id,
            &start_block,
            &end_block,
            start_node_coordinate,
            end_node_coordinate,
            &child_block_group_id,
        );

        let child_block_group_edges =
            BlockGroupEdge::edges_for_block_group(conn, &child_block_group_id);

        let child_edge_ids_by_key = child_block_group_edges
            .iter()
            .map(|augmented_edge| {
                let edge = &augmented_edge.edge;
                (
                    (
                        edge.source_node_id,
                        edge.source_coordinate,
                        edge.source_strand,
                        edge.target_node_id,
                        edge.target_coordinate,
                        edge.target_strand,
                    ),
                    edge.id,
                )
            })
            .collect::<HashMap<_, _>>();

        // The block group method to derive a subgraph creates copies of the nodes from the parent
        // block group, to make it easier to then stitch them together later.  So we need to use the
        // map returned by derive_subgraph to find the edges in the child graph that correspond to
        // path edges in the parent graph, and create a new path from the child edges.
        let mut new_path_edge_ids = vec![];
        let new_start_target_node_id = new_node_ids_by_old.get(&start_block.node_id).unwrap();
        let new_start_edge = child_block_group_edges
            .iter()
            .find(|e| {
                is_start_node(e.edge.source_node_id)
                    && e.edge.target_node_id == *new_start_target_node_id
                    && e.edge.target_coordinate == start_node_coordinate
            })
            .unwrap();
        new_path_edge_ids.push(new_start_edge.edge.id);

        for edge in &current_path_edges {
            let new_source_node_id = new_node_ids_by_old.get(&edge.source_node_id);
            let new_target_node_id = new_node_ids_by_old.get(&edge.target_node_id);
            if let Some(new_source_node_id) = new_source_node_id
                && let Some(new_target_node_id) = new_target_node_id
            {
                let key = &(
                    *new_source_node_id,
                    edge.source_coordinate,
                    edge.source_strand,
                    *new_target_node_id,
                    edge.target_coordinate,
                    edge.target_strand,
                );
                let child_edge_id = child_edge_ids_by_key.get(key);
                if let Some(child_edge_id) = child_edge_id {
                    new_path_edge_ids.push(*child_edge_id);
                }
            }
        }

        let new_end_source_node_id = new_node_ids_by_old.get(&end_block.node_id).unwrap();
        let new_end_edge = child_block_group_edges
            .iter()
            .find(|e| {
                is_end_node(e.edge.target_node_id)
                    && e.edge.source_node_id == *new_end_source_node_id
                    && e.edge.source_coordinate == end_node_coordinate
            })
            .unwrap();
        new_path_edge_ids.push(new_end_edge.edge.id);

        Path::create(
            conn,
            &current_path.name,
            &child_block_group_id,
            &new_path_edge_ids,
        );
    }

    let summary_str = format!(
        " {}: {} new derived block group(s)",
        new_sample_name,
        chunk_ranges.len()
    );
    let op = end_operation(
        conn,
        operation_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: "".to_string(),
                file_type: FileTypes::None,
            }],
            description: "derive chunks".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(GraphOperationError::OperationError);

    println!("Derived chunks successfully.");

    op
}

fn get_block_group_id(
    conn: &Connection,
    collection_name: &str,
    parent_sample_name: Option<&str>,
    region_name: &str,
) -> Result<HashId, GraphOperationError> {
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);

    for block_group in &block_groups {
        if block_group.name == region_name {
            return Ok(block_group.id);
        }
    }

    Err(GraphOperationError::RegionNotFound(format!(
        "No region found with name: {region_name}"
    )))
}

pub fn make_stitch(
    conn: &Connection,
    operation_conn: &Connection,
    collection_name: &str,
    parent_sample_name: Option<&str>,
    new_sample_name: &str,
    region_names: &Vec<&str>,
    new_region_name: &str,
) -> Result<Operation, GraphOperationError> {
    let mut session = start_operation(conn);

    let _new_sample = Sample::get_or_create(conn, new_sample_name);
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);

    let mut block_groups_by_name = HashMap::new();
    for block_group in &block_groups {
        let block_group_name = block_group.name.as_str();
        if region_names.contains(&block_group_name) {
            block_groups_by_name.insert(block_group_name, block_group);
        }
    }

    let mut source_node_coordinates: Vec<(HashId, i64, Strand)> = vec![];
    let mut edges_to_reuse = vec![];
    let mut edges_to_create = vec![];
    let mut concatenated_path_edges = vec![];

    // Part 1
    // * Collect all the existing edges from the regions to be stitched together
    // * Except edges to/from terminal nodes
    // * Also build up a list of edges to create to stitch end nodes of one region to start nodes of
    // the next region
    for region_name in region_names {
        if let Some(block_group) = block_groups_by_name.get(region_name) {
            let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group.id);

            let nonterminal_edges = edges
                .iter()
                .filter(|edge| !edge.edge.is_start_edge() && !edge.edge.is_end_edge())
                .cloned();
            edges_to_reuse.extend(nonterminal_edges);

            let start_edges = edges
                .iter()
                .filter(|edge| edge.edge.is_start_edge())
                .collect::<Vec<_>>();
            // Add all edges between the end nodes of the previous region and the start nodes of
            // this region
            for source_node_coordinate in &source_node_coordinates {
                for start_edge in &start_edges {
                    edges_to_create.push(EdgeData {
                        source_node_id: source_node_coordinate.0,
                        source_coordinate: source_node_coordinate.1,
                        source_strand: source_node_coordinate.2,
                        target_node_id: start_edge.edge.target_node_id,
                        target_coordinate: start_edge.edge.target_coordinate,
                        target_strand: start_edge.edge.target_strand,
                    });
                }
            }

            let end_edges = edges.iter().filter(|edge| edge.edge.is_end_edge());
            source_node_coordinates = end_edges
                .map(|edge| {
                    (
                        edge.edge.source_node_id,
                        edge.edge.source_coordinate,
                        edge.edge.source_strand,
                    )
                })
                .collect();

            let current_path = BlockGroup::get_current_path(conn, &block_group.id);
            concatenated_path_edges.extend(PathEdge::edges_for_path(conn, &current_path.id));
        } else {
            return Err(GraphOperationError::RegionNotFound(format!(
                "No region found with name: {region_name}"
            )));
        }
    }

    // Part 2:
    // * Add in existing edges from the virtual start node to the start nodes of the first region
    // * Add in existing edges from the end nodes of the last region to the virtual end node
    let start_region = block_groups_by_name.get(region_names[0]).unwrap();
    let start_region_edges = BlockGroupEdge::edges_for_block_group(conn, &start_region.id);
    for start_region_edge in &start_region_edges {
        if start_region_edge.edge.is_start_edge() {
            edges_to_reuse.push(start_region_edge.clone());
        }
    }

    let end_region = block_groups_by_name
        .get(region_names[region_names.len() - 1])
        .unwrap();
    let end_region_edges = BlockGroupEdge::edges_for_block_group(conn, &end_region.id);
    for end_region_edge in &end_region_edges {
        if end_region_edge.edge.is_end_edge() {
            edges_to_reuse.push(end_region_edge.clone());
        }
    }

    // Part 3: Set up the block group, set up bg edges for the edges to reuse, create the necessary
    // new edges.
    // We'll do a bulk create for the bg edges later in one big call, once we have more information
    // for the new edges (which will also get bg edges created then)
    let child_block_group = BlockGroup::create(
        conn,
        collection_name,
        Some(new_sample_name),
        new_region_name,
    );

    let mut bg_edges = edges_to_reuse
        .iter()
        .map(|edge| BlockGroupEdgeData {
            block_group_id: child_block_group.id,
            edge_id: edge.edge.id,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        })
        .collect::<Vec<_>>();

    let created_edge_ids = Edge::bulk_create(conn, &edges_to_create);
    let created_edges = Edge::query_by_ids(conn, &created_edge_ids);

    // Part 4: Set up a new path
    let created_edges_by_node_info = created_edges
        .iter()
        .map(|edge| {
            (
                (
                    edge.source_node_id,
                    edge.source_coordinate,
                    edge.source_strand,
                    edge.target_node_id,
                    edge.target_coordinate,
                    edge.target_strand,
                ),
                edge.clone(),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut stitch_path_edge_ids = vec![];
    for (path_edge1, path_edge2) in concatenated_path_edges.iter().tuple_windows() {
        if is_end_node(path_edge1.target_node_id) && is_start_node(path_edge2.source_node_id) {
            stitch_path_edge_ids.push(
                created_edges_by_node_info[&(
                    path_edge1.source_node_id,
                    path_edge1.source_coordinate,
                    path_edge1.source_strand,
                    path_edge2.target_node_id,
                    path_edge2.target_coordinate,
                    path_edge2.target_strand,
                )]
                    .id,
            );
        }
    }

    let mut new_path_edge_ids = vec![concatenated_path_edges[0].id];
    let mut stitch_count = 0;
    for path_edge in &concatenated_path_edges {
        if path_edge.is_end_edge() {
            if stitch_count < stitch_path_edge_ids.len() {
                new_path_edge_ids.push(stitch_path_edge_ids[stitch_count]);
                stitch_count += 1;
            }
        } else if !path_edge.is_start_edge() {
            new_path_edge_ids.push(path_edge.id);
        }
    }

    new_path_edge_ids.push(concatenated_path_edges[concatenated_path_edges.len() - 1].id);

    // Part 5: Create bg edges for the new edges
    let mut chromosome_index_counter = edges_to_reuse
        .iter()
        .max_by(|x, y| x.chromosome_index.cmp(&y.chromosome_index))
        .unwrap()
        .chromosome_index
        + 1;

    let path_edge_id_set = new_path_edge_ids.iter().collect::<HashSet<_>>();
    for created_edge in created_edges {
        if path_edge_id_set.contains(&created_edge.id) {
            bg_edges.push(BlockGroupEdgeData {
                block_group_id: child_block_group.id,
                edge_id: created_edge.id,
                chromosome_index: 0,
                phased: 0,
            });
        } else {
            bg_edges.push(BlockGroupEdgeData {
                block_group_id: child_block_group.id,
                edge_id: created_edge.id,
                chromosome_index: chromosome_index_counter,
                phased: 0,
            });
            chromosome_index_counter += 1;
        }
    }

    BlockGroupEdge::bulk_create(conn, &bg_edges);

    Path::create(
        conn,
        new_region_name,
        &child_block_group.id,
        &new_path_edge_ids,
    );

    let summary_str = format!(
        " {}: stitched {} chunks into new graph",
        new_sample_name,
        region_names.len()
    );

    end_operation(
        conn,
        operation_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: "".to_string(),
                file_type: FileTypes::None,
            }],
            description: "make stitch".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(GraphOperationError::OperationError)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::Strand;
    use gen_models::{
        block_group_edge::BlockGroupEdgeData, collection::Collection, edge::Edge, node::Node,
        operations::setup_db, sequence::Sequence,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{
            get_connection, get_operation_connection, setup_block_group, setup_gen_dir,
        },
        track_database,
        updates::fasta::update_with_fasta,
    };

    #[test]
    fn test_derive_chunks_one_insertion() {
        /*
        AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -> GGGGGGGGGG
                          \-> AAAAAAAA ->/
        Subgraph range:  |-----------------|
        Sequences of the subgraph are TAAAAAAAAC, TTTTTCCCCC
         */
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        Collection::create(conn, "test");
        let (block_group1_id, original_path) = setup_block_group(conn);

        let intervaltree = original_path.intervaltree(conn);
        let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
        let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn);
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str(&format!("test-insert-a.{}", insert_sequence.hash)),
        );
        let edge_into_insert = Edge::create(
            conn,
            insert_start_node_id,
            6,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        );
        let edge_out_of_insert = Edge::create(
            conn,
            insert_node_id,
            8,
            Strand::Forward,
            insert_end_node_id,
            4,
            Strand::Forward,
        );
        let ref_heal_1 = Edge::create(
            conn,
            insert_start_node_id,
            6,
            Strand::Forward,
            insert_start_node_id,
            6,
            Strand::Forward,
        );
        let ref_heal_2 = Edge::create(
            conn,
            insert_end_node_id,
            4,
            Strand::Forward,
            insert_end_node_id,
            4,
            Strand::Forward,
        );

        let edge_ids = [
            edge_into_insert.id,
            edge_out_of_insert.id,
            ref_heal_1.id,
            ref_heal_2.id,
        ];
        let block_group_edges = edge_ids
            .iter()
            .enumerate()
            .map(|(i, edge_id)| BlockGroupEdgeData {
                block_group_id: block_group1_id,
                edge_id: *edge_id,
                chromosome_index: if i < 2 { 1 } else { 0 },
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let insert_path =
            original_path.new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert);
        assert_eq!(
            insert_path.sequence(conn),
            "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
        );

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group1_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        derive_chunks(
            conn,
            op_conn,
            "test",
            None,
            "test",
            "chr1",
            None,
            vec![Range { start: 15, end: 25 }],
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, "test", Some("test"));
        let block_group2 = block_groups.iter().find(|x| x.name == "chr1").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TTTTTCCCCC".to_string(), "TAAAAAAAAC".to_string(),])
        );

        let new_path = BlockGroup::get_current_path(conn, &block_group2.id);
        assert_eq!(new_path.sequence(conn), "TAAAAAAAAC");
    }

    #[test]
    fn derive_chunks_two_inserts() {
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aa.fa");

        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test";

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        let _ = update_with_fasta(
            conn,
            op_conn,
            collection,
            None,
            "test1",
            "m123",
            3,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let _ = update_with_fasta(
            conn,
            op_conn,
            collection,
            Some("test1"),
            "test2",
            "m123",
            15,
            20,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let original_block_groups = Sample::get_block_groups(conn, collection, None);
        let original_block_group_id = &original_block_groups[0].id;
        let all_original_sequences =
            BlockGroup::get_all_sequences(conn, original_block_group_id, false);
        assert_eq!(
            all_original_sequences,
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );

        let grandchild_block_groups = Sample::get_block_groups(conn, collection, Some("test2"));
        let grandchild_block_group_id = &grandchild_block_groups[0].id;
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, grandchild_block_group_id, false);
        assert_eq!(
            all_grandchild_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCAAGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCAAGGAACACACAGAGA".to_string(),
            ])
        );

        derive_chunks(
            conn,
            op_conn,
            collection,
            Some("test2"),
            "test3",
            "m123",
            None,
            vec![
                Range { start: 0, end: 1 },
                Range { start: 1, end: 8 },
                Range { start: 8, end: 25 },
                Range { start: 25, end: 31 },
            ],
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, Some("test3"));
        let block_group2 = block_groups.iter().find(|x| x.name == "m123.2").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TCAATCG".to_string(), "TCGATCG".to_string(),])
        );

        let path2 = BlockGroup::get_current_path(conn, &block_group2.id);
        assert_eq!(path2.sequence(conn), "TCAATCG");

        let block_group3 = block_groups.iter().find(|x| x.name == "m123.3").unwrap();
        let all_sequences3 = BlockGroup::get_all_sequences(conn, &block_group3.id, false);
        assert_eq!(
            all_sequences3,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACA".to_string(),
                "ATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path3 = BlockGroup::get_current_path(conn, &block_group3.id);
        assert_eq!(path3.sequence(conn), "ATCGATCAAGGAACACA");
    }

    #[test]
    fn derive_chunks_two_inserts_then_stitch() {
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aa.fa");

        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test";

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        let _ = update_with_fasta(
            conn,
            op_conn,
            collection,
            None,
            "test1",
            "m123",
            3,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let _ = update_with_fasta(
            conn,
            op_conn,
            collection,
            Some("test1"),
            "test2",
            "m123",
            15,
            20,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let original_block_groups = Sample::get_block_groups(conn, collection, None);
        let original_block_group_id = &original_block_groups[0].id;
        let all_original_sequences =
            BlockGroup::get_all_sequences(conn, original_block_group_id, false);
        assert_eq!(
            all_original_sequences,
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );

        let grandchild_block_groups = Sample::get_block_groups(conn, collection, Some("test2"));
        let grandchild_block_group_id = &grandchild_block_groups[0].id;
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, grandchild_block_group_id, false);
        assert_eq!(
            all_grandchild_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCAAGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCAAGGAACACACAGAGA".to_string(),
            ])
        );

        derive_chunks(
            conn,
            op_conn,
            collection,
            Some("test2"),
            "test3",
            "m123",
            None,
            vec![
                Range { start: 0, end: 1 },
                Range { start: 1, end: 8 },
                Range { start: 8, end: 25 },
                Range { start: 25, end: 31 },
            ],
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, Some("test3"));
        let block_group2 = block_groups.iter().find(|x| x.name == "m123.2").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TCAATCG".to_string(), "TCGATCG".to_string(),])
        );

        let path2 = BlockGroup::get_current_path(conn, &block_group2.id);
        assert_eq!(path2.sequence(conn), "TCAATCG");

        let block_group3 = block_groups.iter().find(|x| x.name == "m123.3").unwrap();
        let all_sequences3 = BlockGroup::get_all_sequences(conn, &block_group3.id, false);
        assert_eq!(
            all_sequences3,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACA".to_string(),
                "ATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path3 = BlockGroup::get_current_path(conn, &block_group3.id);
        assert_eq!(path3.sequence(conn), "ATCGATCAAGGAACACA");

        // Stitch the two main chunks back together in same order
        let _res = make_stitch(
            conn,
            op_conn,
            collection,
            Some("test3"),
            "test4",
            &vec!["m123.2", "m123.3"],
            "m123.stitched",
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, Some("test4"));
        let block_group4 = block_groups
            .iter()
            .find(|x| x.name == "m123.stitched")
            .unwrap();

        let all_sequences4 = BlockGroup::get_all_sequences(conn, &block_group4.id, false);
        assert_eq!(
            all_sequences4,
            HashSet::from_iter(vec![
                "TCAATCGATCGATCAAGGAACACA".to_string(),
                "TCAATCGATCGATCGATCGGGAACACA".to_string(),
                "TCGATCGATCGATCAAGGAACACA".to_string(),
                "TCGATCGATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path4 = BlockGroup::get_current_path(conn, &block_group4.id);
        // path2 + path3 concatenated
        assert_eq!(path4.sequence(conn), "TCAATCGATCGATCAAGGAACACA");

        // Stitch the two main chunks together but in reverse order
        make_stitch(
            conn,
            op_conn,
            collection,
            Some("test3"),
            "test5",
            &vec!["m123.3", "m123.2"],
            "m123.reverse-stitched",
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, Some("test5"));
        let block_group5 = block_groups
            .iter()
            .find(|x| x.name == "m123.reverse-stitched")
            .unwrap();

        let all_sequences5 = BlockGroup::get_all_sequences(conn, &block_group5.id, false);
        assert_eq!(
            all_sequences5,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACATCAATCG".to_string(),
                "ATCGATCAAGGAACACATCGATCG".to_string(),
                "ATCGATCGATCGGGAACACATCAATCG".to_string(),
                "ATCGATCGATCGGGAACACATCGATCG".to_string(),
            ])
        );

        let path5 = BlockGroup::get_current_path(conn, &block_group5.id);
        // path3 + path2 concatenated
        assert_eq!(path5.sequence(conn), "ATCGATCAAGGAACACATCAATCG");
    }
}
