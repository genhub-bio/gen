#'
#' @section Methods:
#'\subsection{Method `get_samples`}{
#'All samples in the repository, each holding its sequence graphs.
#'}
#'
#'\subsection{Method `create_graph_controller`}{
#'Builds a persistent, live `RGraphController` for `sequence_graph_id`. The returned
#'controller is mutated in place by its own methods and renders its current state to
#'PNG bytes on request.
#'}
#'
Repository <- new.env(parent = emptyenv())

Repository$new <- function(path) .Call("wrap__Repository__new", path, PACKAGE = "genr")

Repository$gen_dir <- function() .Call("wrap__Repository__gen_dir", self, PACKAGE = "genr")

Repository$db_path <- function() .Call("wrap__Repository__db_path", self, PACKAGE = "genr")

Repository$execute <- function(query) .Call("wrap__Repository__execute", self, query, PACKAGE = "genr")

Repository$query <- function(query) .Call("wrap__Repository__query", self, query, PACKAGE = "genr")

Repository$get_sequence_graph_by_id <- function(id) .Call("wrap__Repository__get_sequence_graph_by_id", self, id, PACKAGE = "genr")

Repository$get_sequence_graphs <- function() .Call("wrap__Repository__get_sequence_graphs", self, PACKAGE = "genr")

Repository$get_sequence_graphs_by_collection <- function(collection_name) .Call("wrap__Repository__get_sequence_graphs_by_collection", self, collection_name, PACKAGE = "genr")

Repository$get_samples <- function() .Call("wrap__Repository__get_samples", self, PACKAGE = "genr")

Repository$get_node_sequence <- function(node_id, sequence_start, sequence_end) .Call("wrap__Repository__get_node_sequence", self, node_id, sequence_start, sequence_end, PACKAGE = "genr")

Repository$import_fasta <- function(filename, sample, collection, fai, gzi) .Call("wrap__Repository__import_fasta", self, filename, sample, collection, fai, gzi, PACKAGE = "genr")

Repository$import_reference_fasta <- function(filename, reference, collection, fai, gzi) .Call("wrap__Repository__import_reference_fasta", self, filename, reference, collection, fai, gzi, PACKAGE = "genr")

Repository$import_sequences <- function(names, sequences, sample, collection) .Call("wrap__Repository__import_sequences", self, names, sequences, sample, collection, PACKAGE = "genr")

Repository$import_genomic_regions <- function(seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection) .Call("wrap__Repository__import_genomic_regions", self, seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection, PACKAGE = "genr")

Repository$import_gfa <- function(filename, sample, collection) .Call("wrap__Repository__import_gfa", self, filename, sample, collection, PACKAGE = "genr")

Repository$import_genbank <- function(filename, sample, collection) .Call("wrap__Repository__import_genbank", self, filename, sample, collection, PACKAGE = "genr")

Repository$import_library_files <- function(library_name, parts, library, sample, collection) .Call("wrap__Repository__import_library_files", self, library_name, parts, library, sample, collection, PACKAGE = "genr")

Repository$import_library <- function(library_name, parts_list, sample, collection) .Call("wrap__Repository__import_library", self, library_name, parts_list, sample, collection, PACKAGE = "genr")

Repository$update_with_fasta <- function(filename, sample, new_sample, region_name, collection) .Call("wrap__Repository__update_with_fasta", self, filename, sample, new_sample, region_name, collection, PACKAGE = "genr")

Repository$update_with_gfa <- function(filename, sample, new_sample, collection) .Call("wrap__Repository__update_with_gfa", self, filename, sample, new_sample, collection, PACKAGE = "genr")

Repository$update_with_gaf <- function(filename, csv, sample, parent_sample, collection) .Call("wrap__Repository__update_with_gaf", self, filename, csv, sample, parent_sample, collection, PACKAGE = "genr")

Repository$update_with_vcf <- function(filename, genotype, sample, reference, in_place, collection) .Call("wrap__Repository__update_with_vcf", self, filename, genotype, sample, reference, in_place, collection, PACKAGE = "genr")

Repository$update_with_genbank <- function(filename, sample, create_missing, collection) .Call("wrap__Repository__update_with_genbank", self, filename, sample, create_missing, collection, PACKAGE = "genr")

Repository$update_with_library_files <- function(sample, new_sample, path_name, library, parts, collection) .Call("wrap__Repository__update_with_library_files", self, sample, new_sample, path_name, library, parts, collection, PACKAGE = "genr")

Repository$update_with_library <- function(sample, new_sample_name, path_name, parts_list, collection) .Call("wrap__Repository__update_with_library", self, sample, new_sample_name, path_name, parts_list, collection, PACKAGE = "genr")

Repository$update_with_sequence <- function(sequence, sample, new_sample, region_name, no_reference_path_update, collection) .Call("wrap__Repository__update_with_sequence", self, sequence, sample, new_sample, region_name, no_reference_path_update, collection, PACKAGE = "genr")

Repository$export_fasta <- function(filename, sample, collection) .Call("wrap__Repository__export_fasta", self, filename, sample, collection, PACKAGE = "genr")

Repository$export_gfa <- function(filename, sample, node_max, collection) .Call("wrap__Repository__export_gfa", self, filename, sample, node_max, collection, PACKAGE = "genr")

Repository$export_genbank <- function(filename, sample, collection) .Call("wrap__Repository__export_genbank", self, filename, sample, collection, PACKAGE = "genr")

Repository$stitch <- function(collection_name, sample_name, new_sample, new_region, regions) .Call("wrap__Repository__stitch", self, collection_name, sample_name, new_sample, new_region, regions, PACKAGE = "genr")

Repository$build_index <- function(sequence_graph_ids, sequence_kind, k) .Call("wrap__Repository__build_index", self, sequence_graph_ids, sequence_kind, k, PACKAGE = "genr")

Repository$search <- function(query, sequence_graph_ids, sequence_kind) .Call("wrap__Repository__search", self, query, sequence_graph_ids, sequence_kind, PACKAGE = "genr")

Repository$clear_index <- function(sequence_graph_ids) .Call("wrap__Repository__clear_index", self, sequence_graph_ids, PACKAGE = "genr")

Repository$derive_subgraph <- function(collection, sample, new_sample, region, backbone) .Call("wrap__Repository__derive_subgraph", self, collection, sample, new_sample, region, backbone, PACKAGE = "genr")

Repository$derive_chunks <- function(collection, sample, new_sample, region, backbone, breakpoints, chunk_size) .Call("wrap__Repository__derive_chunks", self, collection, sample, new_sample, region, backbone, breakpoints, chunk_size, PACKAGE = "genr")

Repository$auto_load_annotation_groups <- function(sequence_graph_id) .Call("wrap__Repository__auto_load_annotation_groups", self, sequence_graph_id, PACKAGE = "genr")

Repository$list_annotations <- function(sequence_graph_id) .Call("wrap__Repository__list_annotations", self, sequence_graph_id, PACKAGE = "genr")

Repository$create_graph_controller <- function(sequence_graph_id, detail) .Call("wrap__Repository__create_graph_controller", self, sequence_graph_id, detail, PACKAGE = "genr")

#' @export
`$.Repository` <- function (self, name) { func <- Repository[[name]]; environment(func) <- environment(); func }

#' @export
`[[.Repository` <- `$.Repository`

#'
#' @section Methods:
#'\subsection{Method `list_annotations`}{
#'List the gene annotations associated with this sequence graph.
#'
#'Reads persisted annotations from the database (including those inherited
#'from ancestor samples), so it does not depend on the viewer/widget.
#'
#' \subsection{return}{
#'A list of `gen_annotation` records, each with `id`, `name`,
#'`group`, `kind`, `segments`, `length`, and `locus` fields.
#'}
#'}
#'
#'\subsection{Method `translate_annotation`}{
#'Translate a sequence graph or annotation into a protein SequenceGraph.
#'
#'When `region` is a character string it is resolved against this sequence
#'graph only, in priority order: a named path within this graph first,
#'then an annotation in this graph's lineage. No other sequence graphs
#'are searched.
#'
#' \subsection{Arguments}{
#'\describe{
#'\item{`region`}{One of: `NULL` to translate the entire sequence graph; a path name or annotation name scoped to this sequence graph (path names take priority); or a `gen_annotation` record from `list_annotations()` (matched by database id, so unambiguous).}
#'\item{`start`}{0-based path-space coordinate to translate from. Defaults to 0 (the start of the path) when NULL, and is ignored when `region` names an annotation (the annotation's own entry point is used instead). Translation reads forward from this coordinate to its own first in-frame stop codon; it is not bounded by any end coordinate. Default: NULL.}
#'\item{`output_collection`}{Collection for the protein sequence graph. Defaults to this graph's collection.}
#'\item{`name`}{Name for the protein sequence graph. Defaults to "{region} (protein)".}
#'\item{`strand`}{`"forward"` or `"reverse"`. NULL infers from the annotation.}
#'\item{`frame`}{Initial reading frame offset: 0, 1, or 2.}
#'\item{`codon_table`}{NCBI codon table ID (default: 1 = Standard).}
#'}}
#' \subsection{return}{
#'A new SequenceGraph containing the protein sequence, in this
#'graph's sample.
#'}
#'}
#'
SequenceGraph <- new.env(parent = emptyenv())

SequenceGraph$id <- function() .Call("wrap__SequenceGraph__id", self, PACKAGE = "genr")

SequenceGraph$collection <- function() .Call("wrap__SequenceGraph__collection", self, PACKAGE = "genr")

SequenceGraph$sample_name <- function() .Call("wrap__SequenceGraph__sample_name", self, PACKAGE = "genr")

SequenceGraph$name <- function() .Call("wrap__SequenceGraph__name", self, PACKAGE = "genr")

SequenceGraph$db_path <- function() .Call("wrap__SequenceGraph__db_path", self, PACKAGE = "genr")

SequenceGraph$gen_dir <- function() .Call("wrap__SequenceGraph__gen_dir", self, PACKAGE = "genr")

SequenceGraph$export_fasta <- function(filename) .Call("wrap__SequenceGraph__export_fasta", self, filename, PACKAGE = "genr")

SequenceGraph$export_gfa <- function(filename, node_max) .Call("wrap__SequenceGraph__export_gfa", self, filename, node_max, PACKAGE = "genr")

SequenceGraph$export_genbank <- function(filename) .Call("wrap__SequenceGraph__export_genbank", self, filename, PACKAGE = "genr")

SequenceGraph$build_index <- function(sequence_kind, k) .Call("wrap__SequenceGraph__build_index", self, sequence_kind, k, PACKAGE = "genr")

SequenceGraph$search <- function(query, sequence_kind) .Call("wrap__SequenceGraph__search", self, query, sequence_kind, PACKAGE = "genr")

SequenceGraph$clear_index <- function() .Call("wrap__SequenceGraph__clear_index", self, PACKAGE = "genr")

SequenceGraph$get_node_sequence <- function(node_id, sequence_start, sequence_end) .Call("wrap__SequenceGraph__get_node_sequence", self, node_id, sequence_start, sequence_end, PACKAGE = "genr")

SequenceGraph$subgraph <- function(new_sample, start, end, backbone) .Call("wrap__SequenceGraph__subgraph", self, new_sample, start, end, backbone, PACKAGE = "genr")

SequenceGraph$chunks <- function(new_sample, breakpoints, chunk_size, backbone) .Call("wrap__SequenceGraph__chunks", self, new_sample, breakpoints, chunk_size, backbone, PACKAGE = "genr")

SequenceGraph$to_dict <- function() .Call("wrap__SequenceGraph__to_dict", self, PACKAGE = "genr")

SequenceGraph$list_annotations <- function() .Call("wrap__SequenceGraph__list_annotations", self, PACKAGE = "genr")

SequenceGraph$translate_annotation <- function(region, start, output_collection, name, strand, frame, codon_table) .Call("wrap__SequenceGraph__translate_annotation", self, region, start, output_collection, name, strand, frame, codon_table, PACKAGE = "genr")

#' @export
`$.SequenceGraph` <- function (self, name) { func <- SequenceGraph[[name]]; environment(func) <- environment(); func }

#' @export
`[[.SequenceGraph` <- `$.SequenceGraph`

#' A persistent, mutable-in-place graph controller for the R viewers. Each method mutates the
#' shared graph session directly and rendering draws its current state to PNG bytes, so the
#' R widget shows the same pixels as the Python widget and holds no operation history.
#'
#' @section Methods:
#'\subsection{Method `go_to_match`}{
#'Move the viewport to the base `frac_x` of the way along the block `node_id`
#'`seq_start..seq_end`, snapping it left unless `center` is set. The zoom level stays as
#'the user set it.
#'}
#'
#'\subsection{Method `clear_highlights`}{
#'Remove match highlights, keeping named tracks and the selected path.
#'}
#'
#'\subsection{Method `trigger_auto_load`}{
#'Show every annotation group in the automatic theme palette.
#'}
#'
#'\subsection{Method `load_annotation_groups_with_colors_json`}{
#'Show every annotation group, coloring annotations from a JSON object mapping annotation
#'id hex strings to a `#rrggbb` color or `null` to hide that annotation.
#'}
#'
#'\subsection{Method `render_png`}{
#'Render the graph alone to PNG bytes, cropped to its content, with `scale` image pixels
#'per logical pixel.
#'}
#'
#'\subsection{Method `render_png_viewport`}{
#'Render the full interactive viewport to PNG bytes, keeping the header and every cell at
#'its position so pointer coordinates map straight to cells.
#'}
#'
#'\subsection{Method `png_cell_size`}{
#'The pixel size of one terminal cell in a PNG rendered at `scale`, as `c(width, height)`.
#'}
#'
#'\subsection{Method `background_color`}{
#'The canvas color of the PNG renderer, as `#rrggbb`.
#'}
#'
#'\subsection{Method `hit_regions`}{
#'The clickable cells of the last rendered viewport, as a JSON array of
#'`{"x", "y", "kind"}` with `kind` either `"node"` or `"wormhole"`.
#'}
#'
RGraphController <- new.env(parent = emptyenv())

RGraphController$set_detail <- function(detail) .Call("wrap__RGraphController__set_detail", self, detail, PACKAGE = "genr")

RGraphController$zoom_in <- function() .Call("wrap__RGraphController__zoom_in", self, PACKAGE = "genr")

RGraphController$zoom_out <- function() .Call("wrap__RGraphController__zoom_out", self, PACKAGE = "genr")

RGraphController$move_by <- function(dx, dy) .Call("wrap__RGraphController__move_by", self, dx, dy, PACKAGE = "genr")

RGraphController$handle_click <- function(col, row) .Call("wrap__RGraphController__handle_click", self, col, row, PACKAGE = "genr")

RGraphController$go_to_match <- function(node_id, seq_start, seq_end, frac_x, center) .Call("wrap__RGraphController__go_to_match", self, node_id, seq_start, seq_end, frac_x, center, PACKAGE = "genr")

RGraphController$highlight_match <- function(color, start_offset, end_offset, strand, node_ids, seq_starts, seq_ends) .Call("wrap__RGraphController__highlight_match", self, color, start_offset, end_offset, strand, node_ids, seq_starts, seq_ends, PACKAGE = "genr")

RGraphController$clear_highlights <- function() .Call("wrap__RGraphController__clear_highlights", self, PACKAGE = "genr")

RGraphController$add_track_group <- function(group) .Call("wrap__RGraphController__add_track_group", self, group, PACKAGE = "genr")

RGraphController$add_track_file <- function(path, name, sample) .Call("wrap__RGraphController__add_track_file", self, path, name, sample, PACKAGE = "genr")

RGraphController$trigger_auto_load <- function() .Call("wrap__RGraphController__trigger_auto_load", self, PACKAGE = "genr")

RGraphController$load_annotation_groups_with_colors_json <- function(color_map_json) .Call("wrap__RGraphController__load_annotation_groups_with_colors_json", self, color_map_json, PACKAGE = "genr")

RGraphController$render_png <- function(cols, rows, scale) .Call("wrap__RGraphController__render_png", self, cols, rows, scale, PACKAGE = "genr")

RGraphController$render_png_viewport <- function(cols, rows, scale) .Call("wrap__RGraphController__render_png_viewport", self, cols, rows, scale, PACKAGE = "genr")

RGraphController$png_cell_size <- function(scale) .Call("wrap__RGraphController__png_cell_size", self, scale, PACKAGE = "genr")

RGraphController$background_color <- function() .Call("wrap__RGraphController__background_color", self, PACKAGE = "genr")

RGraphController$hit_regions <- function(cols, rows) .Call("wrap__RGraphController__hit_regions", self, cols, rows, PACKAGE = "genr")

RGraphController$clone_controller <- function() .Call("wrap__RGraphController__clone_controller", self, PACKAGE = "genr")

#' @export
`$.RGraphController` <- function (self, name) { func <- RGraphController[[name]]; environment(func) <- environment(); func }

#' @export
`[[.RGraphController` <- `$.RGraphController`


