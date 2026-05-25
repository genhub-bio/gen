#' Initialise a Gen workspace in the current directory.
#' @export
init <- function() .Call("wrap__init", PACKAGE = "genr")

#' Return the path to the current workspace's .gen directory.
#' @export
get_gen_dir <- function() .Call("wrap__get_gen_dir", PACKAGE = "genr")

#' Open a Gen database context.
#' @export
db_context <- function(workspace_path, db_path) .Call("wrap__db_context", workspace_path, db_path, PACKAGE = "genr")

import_fasta <- function(workspace_path, db_path, filename, sample, shallow, collection) .Call("wrap__import_fasta", workspace_path, db_path, filename, sample, shallow, collection, PACKAGE = "genr")

import_sequences <- function(workspace_path, db_path, names, sequences, sample, collection) .Call("wrap__import_sequences", workspace_path, db_path, names, sequences, sample, collection, PACKAGE = "genr")

import_genomic_regions <- function(workspace_path, db_path, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection) .Call("wrap__import_genomic_regions", workspace_path, db_path, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection, PACKAGE = "genr")

import_gfa <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__import_gfa", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

import_genbank <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__import_genbank", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

import_library_files <- function(workspace_path, db_path, library_name, parts, library, sample, collection) .Call("wrap__import_library_files", workspace_path, db_path, library_name, parts, library, sample, collection, PACKAGE = "genr")

import_library <- function(workspace_path, db_path, library_name, parts_list, sample, collection) .Call("wrap__import_library", workspace_path, db_path, library_name, parts_list, sample, collection, PACKAGE = "genr")

update_with_fasta <- function(workspace_path, db_path, filename, sample, new_sample, region_name, collection) .Call("wrap__update_with_fasta", workspace_path, db_path, filename, sample, new_sample, region_name, collection, PACKAGE = "genr")

update_with_gfa <- function(workspace_path, db_path, filename, sample, new_sample, collection) .Call("wrap__update_with_gfa", workspace_path, db_path, filename, sample, new_sample, collection, PACKAGE = "genr")

update_with_gaf <- function(workspace_path, db_path, filename, csv, sample, parent_sample, collection) .Call("wrap__update_with_gaf", workspace_path, db_path, filename, csv, sample, parent_sample, collection, PACKAGE = "genr")

update_with_vcf <- function(workspace_path, db_path, filename, genotype, sample, parent_samples, in_place, collection) .Call("wrap__update_with_vcf", workspace_path, db_path, filename, genotype, sample, parent_samples, in_place, collection, PACKAGE = "genr")

update_with_genbank <- function(workspace_path, db_path, filename, sample, create_missing, collection) .Call("wrap__update_with_genbank", workspace_path, db_path, filename, sample, create_missing, collection, PACKAGE = "genr")

update_with_library_files <- function(workspace_path, db_path, sample, new_sample, path_name, library, parts, collection) .Call("wrap__update_with_library_files", workspace_path, db_path, sample, new_sample, path_name, library, parts, collection, PACKAGE = "genr")

update_with_library <- function(workspace_path, db_path, sample, new_sample_name, path_name, parts_list, collection) .Call("wrap__update_with_library", workspace_path, db_path, sample, new_sample_name, path_name, parts_list, collection, PACKAGE = "genr")

update_with_sequence <- function(workspace_path, db_path, sequence, sample, new_sample, region_name, no_reference_path_update, collection) .Call("wrap__update_with_sequence", workspace_path, db_path, sequence, sample, new_sample, region_name, no_reference_path_update, collection, PACKAGE = "genr")

export_fasta <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__export_fasta", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

export_gfa <- function(workspace_path, db_path, filename, sample, node_max, collection) .Call("wrap__export_gfa", workspace_path, db_path, filename, sample, node_max, collection, PACKAGE = "genr")

export_genbank <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__export_genbank", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

derive_chunks <- function(workspace_path, db_path, sample, new_sample, region, backbone, breakpoints, chunk_size, collection) .Call("wrap__derive_chunks", workspace_path, db_path, sample, new_sample, region, backbone, breakpoints, chunk_size, collection, PACKAGE = "genr")

derive_subgraph <- function(workspace_path, db_path, sample, new_sample, region, backbone, collection) .Call("wrap__derive_subgraph", workspace_path, db_path, sample, new_sample, region, backbone, collection, PACKAGE = "genr")

repo_get_gen_dir <- function(path) .Call("wrap__repo_get_gen_dir", path, PACKAGE = "genr")

repo_get_db_path <- function(path) .Call("wrap__repo_get_db_path", path, PACKAGE = "genr")

repo_execute <- function(db_path, query) .Call("wrap__repo_execute", db_path, query, PACKAGE = "genr")

repo_query <- function(db_path, query) .Call("wrap__repo_query", db_path, query, PACKAGE = "genr")

repo_get_block_group_by_id <- function(db_path, id) .Call("wrap__repo_get_block_group_by_id", db_path, id, PACKAGE = "genr")

repo_get_block_groups <- function(db_path) .Call("wrap__repo_get_block_groups", db_path, PACKAGE = "genr")

repo_get_block_groups_by_collection <- function(db_path, collection_name) .Call("wrap__repo_get_block_groups_by_collection", db_path, collection_name, PACKAGE = "genr")

repo_block_group_to_dict <- function(db_path, block_group_id) .Call("wrap__repo_block_group_to_dict", db_path, block_group_id, PACKAGE = "genr")

repo_get_block_sequence <- function(db_path, node_id, sequence_start, sequence_end) .Call("wrap__repo_get_block_sequence", db_path, node_id, sequence_start, sequence_end, PACKAGE = "genr")

repo_stitch <- function(workspace_path, db_path, collection_name, sample_name, new_sample, new_region, regions) .Call("wrap__repo_stitch", workspace_path, db_path, collection_name, sample_name, new_sample, new_region, regions, PACKAGE = "genr")

repo_build_index <- function(db_path, gen_dir, block_group_ids, sequence_kind, k) .Call("wrap__repo_build_index", db_path, gen_dir, block_group_ids, sequence_kind, k, PACKAGE = "genr")

repo_search <- function(db_path, gen_dir, query, block_group_ids, sequence_kind) .Call("wrap__repo_search", db_path, gen_dir, query, block_group_ids, sequence_kind, PACKAGE = "genr")

repo_clear_index <- function(gen_dir, block_group_ids) .Call("wrap__repo_clear_index", gen_dir, block_group_ids, PACKAGE = "genr")

repo_bg_subgraph <- function(workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, start, end, backbone) .Call("wrap__repo_bg_subgraph", workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, start, end, backbone, PACKAGE = "genr")

repo_bg_chunks <- function(workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, breakpoints, chunk_size, backbone) .Call("wrap__repo_bg_chunks", workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, breakpoints, chunk_size, backbone, PACKAGE = "genr")

repo_bg_export_fasta <- function(db_path, collection_name, sample_name, filename) .Call("wrap__repo_bg_export_fasta", db_path, collection_name, sample_name, filename, PACKAGE = "genr")

repo_bg_export_gfa <- function(db_path, collection_name, sample_name, filename, node_max) .Call("wrap__repo_bg_export_gfa", db_path, collection_name, sample_name, filename, node_max, PACKAGE = "genr")

repo_bg_export_genbank <- function(db_path, collection_name, sample_name, filename) .Call("wrap__repo_bg_export_genbank", db_path, collection_name, sample_name, filename, PACKAGE = "genr")

graph_render_frame <- function(db_path, block_group_id, detail, cols, rows, ops, tracks_json) .Call("wrap__graph_render_frame", db_path, block_group_id, detail, cols, rows, ops, tracks_json, PACKAGE = "genr")

graph_handle_click <- function(db_path, block_group_id, detail, ops, col, row) .Call("wrap__graph_handle_click", db_path, block_group_id, detail, ops, col, row, PACKAGE = "genr")


