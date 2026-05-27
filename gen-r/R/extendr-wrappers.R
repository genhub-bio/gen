#' Open a Gen database context.
db_context <- function(workspace_path, db_path) .Call("wrap__db_context", workspace_path, db_path, PACKAGE = "genr")

import_fasta <- function(workspace_path, db_path, filename, sample, shallow, collection) .Call("wrap__import_fasta", workspace_path, db_path, filename, sample, shallow, collection, PACKAGE = "genr")

import_reference_fasta <- function(workspace_path, db_path, filename, reference, shallow, collection) .Call("wrap__import_reference_fasta", workspace_path, db_path, filename, reference, shallow, collection, PACKAGE = "genr")

import_sequences <- function(workspace_path, db_path, names, sequences, sample, collection) .Call("wrap__import_sequences", workspace_path, db_path, names, sequences, sample, collection, PACKAGE = "genr")

import_genomic_regions <- function(workspace_path, db_path, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection) .Call("wrap__import_genomic_regions", workspace_path, db_path, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection, PACKAGE = "genr")

import_gfa <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__import_gfa", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

import_genbank <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__import_genbank", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

import_library_files <- function(workspace_path, db_path, library_name, parts, library, sample, collection) .Call("wrap__import_library_files", workspace_path, db_path, library_name, parts, library, sample, collection, PACKAGE = "genr")

import_library <- function(workspace_path, db_path, library_name, parts_list, sample, collection) .Call("wrap__import_library", workspace_path, db_path, library_name, parts_list, sample, collection, PACKAGE = "genr")

update_with_fasta <- function(workspace_path, db_path, filename, sample, new_sample, region_name, collection) .Call("wrap__update_with_fasta", workspace_path, db_path, filename, sample, new_sample, region_name, collection, PACKAGE = "genr")

update_with_gfa <- function(workspace_path, db_path, filename, sample, new_sample, collection) .Call("wrap__update_with_gfa", workspace_path, db_path, filename, sample, new_sample, collection, PACKAGE = "genr")

update_with_gaf <- function(workspace_path, db_path, filename, csv, sample, parent_sample, collection) .Call("wrap__update_with_gaf", workspace_path, db_path, filename, csv, sample, parent_sample, collection, PACKAGE = "genr")

update_with_vcf <- function(workspace_path, db_path, filename, genotype, sample, reference, in_place, collection) .Call("wrap__update_with_vcf", workspace_path, db_path, filename, genotype, sample, reference, in_place, collection, PACKAGE = "genr")

update_with_genbank <- function(workspace_path, db_path, filename, sample, create_missing, collection) .Call("wrap__update_with_genbank", workspace_path, db_path, filename, sample, create_missing, collection, PACKAGE = "genr")

update_with_library_files <- function(workspace_path, db_path, sample, new_sample, path_name, library, parts, collection) .Call("wrap__update_with_library_files", workspace_path, db_path, sample, new_sample, path_name, library, parts, collection, PACKAGE = "genr")

update_with_library <- function(workspace_path, db_path, sample, new_sample_name, path_name, parts_list, collection) .Call("wrap__update_with_library", workspace_path, db_path, sample, new_sample_name, path_name, parts_list, collection, PACKAGE = "genr")

update_with_sequence <- function(workspace_path, db_path, sequence, sample, new_sample, region_name, no_reference_path_update, collection) .Call("wrap__update_with_sequence", workspace_path, db_path, sequence, sample, new_sample, region_name, no_reference_path_update, collection, PACKAGE = "genr")

export_fasta <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__export_fasta", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

export_gfa <- function(workspace_path, db_path, filename, sample, node_max, collection) .Call("wrap__export_gfa", workspace_path, db_path, filename, sample, node_max, collection, PACKAGE = "genr")

export_genbank <- function(workspace_path, db_path, filename, sample, collection) .Call("wrap__export_genbank", workspace_path, db_path, filename, sample, collection, PACKAGE = "genr")

derive_chunks <- function(workspace_path, db_path, sample, new_sample, region, backbone, breakpoints, chunk_size, collection) .Call("wrap__derive_chunks", workspace_path, db_path, sample, new_sample, region, backbone, breakpoints, chunk_size, collection, PACKAGE = "genr")

derive_subgraph <- function(workspace_path, db_path, sample, new_sample, region, backbone, collection) .Call("wrap__derive_subgraph", workspace_path, db_path, sample, new_sample, region, backbone, collection, PACKAGE = "genr")

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

GenRepository <- new.env(parent = emptyenv())

GenRepository$new <- function(path) .Call("wrap__GenRepository__new", path, PACKAGE = "genr")

GenRepository$gen_dir <- function() .Call("wrap__GenRepository__gen_dir", self, PACKAGE = "genr")

GenRepository$db_path <- function() .Call("wrap__GenRepository__db_path", self, PACKAGE = "genr")

GenRepository$execute <- function(query) .Call("wrap__GenRepository__execute", self, query, PACKAGE = "genr")

GenRepository$query <- function(query) .Call("wrap__GenRepository__query", self, query, PACKAGE = "genr")

GenRepository$get_block_group_by_id <- function(id) .Call("wrap__GenRepository__get_block_group_by_id", self, id, PACKAGE = "genr")

GenRepository$get_block_groups <- function() .Call("wrap__GenRepository__get_block_groups", self, PACKAGE = "genr")

GenRepository$get_block_groups_by_collection <- function(collection_name) .Call("wrap__GenRepository__get_block_groups_by_collection", self, collection_name, PACKAGE = "genr")

GenRepository$block_group_to_dict <- function(block_group_id) .Call("wrap__GenRepository__block_group_to_dict", self, block_group_id, PACKAGE = "genr")

GenRepository$get_block_sequence <- function(node_id, sequence_start, sequence_end) .Call("wrap__GenRepository__get_block_sequence", self, node_id, sequence_start, sequence_end, PACKAGE = "genr")

GenRepository$import_fasta <- function(filename, sample, shallow, collection) .Call("wrap__GenRepository__import_fasta", self, filename, sample, shallow, collection, PACKAGE = "genr")

GenRepository$import_reference_fasta <- function(filename, reference, shallow, collection) .Call("wrap__GenRepository__import_reference_fasta", self, filename, reference, shallow, collection, PACKAGE = "genr")

GenRepository$import_sequences <- function(names, sequences, sample, collection) .Call("wrap__GenRepository__import_sequences", self, names, sequences, sample, collection, PACKAGE = "genr")

GenRepository$import_genomic_regions <- function(seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection) .Call("wrap__GenRepository__import_genomic_regions", self, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection, PACKAGE = "genr")

GenRepository$import_gfa <- function(filename, sample, collection) .Call("wrap__GenRepository__import_gfa", self, filename, sample, collection, PACKAGE = "genr")

GenRepository$import_genbank <- function(filename, sample, collection) .Call("wrap__GenRepository__import_genbank", self, filename, sample, collection, PACKAGE = "genr")

GenRepository$import_library_files <- function(library_name, parts, library, sample, collection) .Call("wrap__GenRepository__import_library_files", self, library_name, parts, library, sample, collection, PACKAGE = "genr")

GenRepository$import_library <- function(library_name, parts_list, sample, collection) .Call("wrap__GenRepository__import_library", self, library_name, parts_list, sample, collection, PACKAGE = "genr")

GenRepository$update_with_fasta <- function(filename, sample, new_sample, region_name, collection) .Call("wrap__GenRepository__update_with_fasta", self, filename, sample, new_sample, region_name, collection, PACKAGE = "genr")

GenRepository$update_with_gfa <- function(filename, sample, new_sample, collection) .Call("wrap__GenRepository__update_with_gfa", self, filename, sample, new_sample, collection, PACKAGE = "genr")

GenRepository$update_with_gaf <- function(filename, csv, sample, parent_sample, collection) .Call("wrap__GenRepository__update_with_gaf", self, filename, csv, sample, parent_sample, collection, PACKAGE = "genr")

GenRepository$update_with_vcf <- function(filename, genotype, sample, reference, in_place, collection) .Call("wrap__GenRepository__update_with_vcf", self, filename, genotype, sample, reference, in_place, collection, PACKAGE = "genr")

GenRepository$update_with_genbank <- function(filename, sample, create_missing, collection) .Call("wrap__GenRepository__update_with_genbank", self, filename, sample, create_missing, collection, PACKAGE = "genr")

GenRepository$update_with_library_files <- function(sample, new_sample, path_name, library, parts, collection) .Call("wrap__GenRepository__update_with_library_files", self, sample, new_sample, path_name, library, parts, collection, PACKAGE = "genr")

GenRepository$update_with_library <- function(sample, new_sample_name, path_name, parts_list, collection) .Call("wrap__GenRepository__update_with_library", self, sample, new_sample_name, path_name, parts_list, collection, PACKAGE = "genr")

GenRepository$update_with_sequence <- function(sequence, sample, new_sample, region_name, no_reference_path_update, collection) .Call("wrap__GenRepository__update_with_sequence", self, sequence, sample, new_sample, region_name, no_reference_path_update, collection, PACKAGE = "genr")

GenRepository$export_fasta <- function(filename, sample, collection) .Call("wrap__GenRepository__export_fasta", self, filename, sample, collection, PACKAGE = "genr")

GenRepository$export_gfa <- function(filename, sample, node_max, collection) .Call("wrap__GenRepository__export_gfa", self, filename, sample, node_max, collection, PACKAGE = "genr")

GenRepository$export_genbank <- function(filename, sample, collection) .Call("wrap__GenRepository__export_genbank", self, filename, sample, collection, PACKAGE = "genr")

GenRepository$stitch <- function(collection_name, sample_name, new_sample, new_region, regions) .Call("wrap__GenRepository__stitch", self, collection_name, sample_name, new_sample, new_region, regions, PACKAGE = "genr")

GenRepository$build_index <- function(block_group_ids, sequence_kind, k) .Call("wrap__GenRepository__build_index", self, block_group_ids, sequence_kind, k, PACKAGE = "genr")

GenRepository$search <- function(query, block_group_ids, sequence_kind) .Call("wrap__GenRepository__search", self, query, block_group_ids, sequence_kind, PACKAGE = "genr")

GenRepository$clear_index <- function(block_group_ids) .Call("wrap__GenRepository__clear_index", self, block_group_ids, PACKAGE = "genr")

GenRepository$derive_subgraph <- function(collection, sample, new_sample, region, backbone) .Call("wrap__GenRepository__derive_subgraph", self, collection, sample, new_sample, region, backbone, PACKAGE = "genr")

GenRepository$derive_chunks <- function(collection, sample, new_sample, region, backbone, breakpoints, chunk_size) .Call("wrap__GenRepository__derive_chunks", self, collection, sample, new_sample, region, backbone, breakpoints, chunk_size, PACKAGE = "genr")

#' @export
`$.GenRepository` <- function (self, name) { func <- GenRepository[[name]]; environment(func) <- environment(); func }

#' @export
`[[.GenRepository` <- `$.GenRepository`

GenBlockGroup <- new.env(parent = emptyenv())

GenBlockGroup$id <- function() .Call("wrap__GenBlockGroup__id", self, PACKAGE = "genr")

GenBlockGroup$collection <- function() .Call("wrap__GenBlockGroup__collection", self, PACKAGE = "genr")

GenBlockGroup$sample_name <- function() .Call("wrap__GenBlockGroup__sample_name", self, PACKAGE = "genr")

GenBlockGroup$name <- function() .Call("wrap__GenBlockGroup__name", self, PACKAGE = "genr")

GenBlockGroup$db_path <- function() .Call("wrap__GenBlockGroup__db_path", self, PACKAGE = "genr")

GenBlockGroup$gen_dir <- function() .Call("wrap__GenBlockGroup__gen_dir", self, PACKAGE = "genr")

GenBlockGroup$export_fasta <- function(filename) .Call("wrap__GenBlockGroup__export_fasta", self, filename, PACKAGE = "genr")

GenBlockGroup$export_gfa <- function(filename, node_max) .Call("wrap__GenBlockGroup__export_gfa", self, filename, node_max, PACKAGE = "genr")

GenBlockGroup$export_genbank <- function(filename) .Call("wrap__GenBlockGroup__export_genbank", self, filename, PACKAGE = "genr")

GenBlockGroup$build_index <- function(sequence_kind, k) .Call("wrap__GenBlockGroup__build_index", self, sequence_kind, k, PACKAGE = "genr")

GenBlockGroup$search <- function(query, sequence_kind) .Call("wrap__GenBlockGroup__search", self, query, sequence_kind, PACKAGE = "genr")

GenBlockGroup$clear_index <- function() .Call("wrap__GenBlockGroup__clear_index", self, PACKAGE = "genr")

GenBlockGroup$subgraph <- function(new_sample, start, end, backbone) .Call("wrap__GenBlockGroup__subgraph", self, new_sample, start, end, backbone, PACKAGE = "genr")

GenBlockGroup$chunks <- function(new_sample, breakpoints, chunk_size, backbone) .Call("wrap__GenBlockGroup__chunks", self, new_sample, breakpoints, chunk_size, backbone, PACKAGE = "genr")

GenBlockGroup$block_group_to_dict <- function() .Call("wrap__GenBlockGroup__block_group_to_dict", self, PACKAGE = "genr")

#' @export
`$.GenBlockGroup` <- function (self, name) { func <- GenBlockGroup[[name]]; environment(func) <- environment(); func }

#' @export
`[[.GenBlockGroup` <- `$.GenBlockGroup`


