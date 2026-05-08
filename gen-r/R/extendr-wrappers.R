# Generated bindings for the Gen Rust module.

init <- function() .Call(wrap__init)
get_gen_dir <- function() .Call(wrap__get_gen_dir)
db_context <- function(workspace_path = NULL, db_path = NULL) .Call(wrap__db_context, workspace_path, db_path)

import_fasta <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = "sample", shallow = FALSE) {
  .Call(wrap__import_fasta, workspace_path, db_path, filename, name, sample, shallow)
}

import_gfa <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = "sample") {
  .Call(wrap__import_gfa, workspace_path, db_path, filename, name, sample)
}

import_genbank <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = "sample") {
  .Call(wrap__import_genbank, workspace_path, db_path, filename, name, sample)
}

import_library_files <- function(workspace_path = NULL, db_path = NULL, library_name, parts, library, name = NULL, sample = "sample") {
  .Call(wrap__import_library_files, workspace_path, db_path, library_name, parts, library, name, sample)
}

import_library <- function(workspace_path = NULL, db_path = NULL, library_name, parts_list, name = NULL, sample = NULL) {
  .Call(wrap__import_library, workspace_path, db_path, library_name, parts_list, name, sample)
}

update_with_fasta <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample, new_sample, region_name) {
  .Call(wrap__update_with_fasta, workspace_path, db_path, filename, name, sample, new_sample, region_name)
}

update_with_gfa <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample, new_sample) {
  .Call(wrap__update_with_gfa, workspace_path, db_path, filename, name, sample, new_sample)
}

update_with_gaf <- function(workspace_path = NULL, db_path = NULL, filename, csv, name = NULL, sample, parent_sample = NULL) {
  .Call(wrap__update_with_gaf, workspace_path, db_path, filename, csv, name, sample, parent_sample)
}

update_with_vcf <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, genotype = NULL, sample = NULL, parent_samples = character(), in_place = FALSE) {
  .Call(wrap__update_with_vcf, workspace_path, db_path, filename, name, genotype, sample, parent_samples, in_place)
}

update_with_genbank <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample, create_missing = FALSE) {
  .Call(wrap__update_with_genbank, workspace_path, db_path, filename, name, sample, create_missing)
}

update_with_library_files <- function(workspace_path = NULL, db_path = NULL, name = NULL, sample, new_sample, path_name, library, parts) {
  .Call(wrap__update_with_library_files, workspace_path, db_path, name, sample, new_sample, path_name, library, parts)
}

update_with_library <- function(workspace_path = NULL, db_path = NULL, name = NULL, sample = NULL, new_sample_name, path_name, parts_list) {
  .Call(wrap__update_with_library, workspace_path, db_path, name, sample, new_sample_name, path_name, parts_list)
}

update_with_sequence <- function(workspace_path = NULL, db_path = NULL, sequence, name = NULL, sample, new_sample, region_name, no_reference_path_update = FALSE) {
  .Call(wrap__update_with_sequence, workspace_path, db_path, sequence, name, sample, new_sample, region_name, no_reference_path_update)
}

export_fasta <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = NULL) {
  .Call(wrap__export_fasta, workspace_path, db_path, filename, name, sample)
}

export_gfa <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample, node_max = NULL) {
  .Call(wrap__export_gfa, workspace_path, db_path, filename, name, sample, node_max)
}

export_genbank <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample) {
  .Call(wrap__export_genbank, workspace_path, db_path, filename, name, sample)
}

derive_chunks <- function(workspace_path = NULL, db_path = NULL, name = NULL, sample, new_sample, region, backbone = NULL, breakpoints = NULL, chunk_size = NULL) {
  .Call(wrap__derive_chunks, workspace_path, db_path, name, sample, new_sample, region, backbone, breakpoints, chunk_size)
}

derive_subgraph <- function(workspace_path = NULL, db_path = NULL, name = NULL, sample, new_sample, region, backbone = NULL) {
  .Call(wrap__derive_subgraph, workspace_path, db_path, name, sample, new_sample, region, backbone)
}

make_stitch <- function(workspace_path = NULL, db_path = NULL, name = NULL, sample, new_sample, regions, new_region) {
  .Call(wrap__make_stitch, workspace_path, db_path, name, sample, new_sample, regions, new_region)
}

repo_get_gen_dir <- function(path = NULL) .Call(wrap__repo_get_gen_dir, path)
repo_get_db_path <- function(path = NULL) .Call(wrap__repo_get_db_path, path)
repo_execute <- function(db_path, query) .Call(wrap__repo_execute, db_path, query)
repo_query <- function(db_path, query) .Call(wrap__repo_query, db_path, query)
repo_get_block_group_by_id <- function(db_path, id) .Call(wrap__repo_get_block_group_by_id, db_path, id)
repo_get_block_groups <- function(db_path) .Call(wrap__repo_get_block_groups, db_path)
repo_get_block_groups_by_collection <- function(db_path, collection_name) .Call(wrap__repo_get_block_groups_by_collection, db_path, collection_name)
repo_block_group_to_dict <- function(db_path, block_group_id) .Call(wrap__repo_block_group_to_dict, db_path, block_group_id)
repo_get_block_sequence <- function(db_path, node_id, sequence_start, sequence_end) .Call(wrap__repo_get_block_sequence, db_path, node_id, sequence_start, sequence_end)
repo_stitch <- function(workspace_path, db_path, collection_name, sample_name, new_sample, new_region, regions) .Call(wrap__repo_stitch, workspace_path, db_path, collection_name, sample_name, new_sample, new_region, regions)
repo_build_index <- function(db_path, gen_dir, block_group_ids = character(), sequence_kind = "dna", k = 16L) .Call(wrap__repo_build_index, db_path, gen_dir, block_group_ids, sequence_kind, k)
repo_search <- function(db_path, gen_dir, query, block_group_ids = character(), sequence_kind = "dna") .Call(wrap__repo_search, db_path, gen_dir, query, block_group_ids, sequence_kind)
repo_clear_index <- function(gen_dir, block_group_ids = character()) .Call(wrap__repo_clear_index, gen_dir, block_group_ids)
repo_bg_subgraph <- function(workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, start, end, backbone = NULL) .Call(wrap__repo_bg_subgraph, workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, start, end, backbone)
repo_bg_chunks <- function(workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, breakpoints = NULL, chunk_size = NULL, backbone = NULL) .Call(wrap__repo_bg_chunks, workspace_path, db_path, collection_name, sample_name, bg_name, new_sample, breakpoints, chunk_size, backbone)
repo_bg_export_fasta <- function(db_path, collection_name, sample_name, filename) .Call(wrap__repo_bg_export_fasta, db_path, collection_name, sample_name, filename)
repo_bg_export_gfa <- function(db_path, collection_name, sample_name, filename, node_max = NULL) .Call(wrap__repo_bg_export_gfa, db_path, collection_name, sample_name, filename, node_max)
repo_bg_export_genbank <- function(db_path, collection_name, sample_name, filename) .Call(wrap__repo_bg_export_genbank, db_path, collection_name, sample_name, filename)
graph_render_frame <- function(db_path, block_group_id, detail, cols, rows, ops) .Call(wrap__graph_render_frame, db_path, block_group_id, detail, cols, rows, ops)
graph_handle_click <- function(db_path, block_group_id, detail, ops, col, row) .Call(wrap__graph_handle_click, db_path, block_group_id, detail, ops, col, row)
