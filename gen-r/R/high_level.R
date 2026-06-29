#' @useDynLib genr, .registration = TRUE
NULL

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

# Extract the genome string from a sequence container.
# Supported: BSgenome, DNAStringSet (with metadata$genome set), FaFile, TwoBitFile.
.container_genome <- function(container) {
  if (inherits(container, "BSgenome")) {
    return(unique(GenomeInfoDb::genome(GenomeInfoDb::seqinfo(container))))
  }
  meta <- S4Vectors::metadata(container)
  meta[["genome"]] %||% stop(
    "seq_container has no genome metadata. Set it with: metadata(x)$genome <- \"<assembly>\""
  )
}

# Auto-generate coordinate-based names for a GRanges when names() are absent.
.granges_names <- function(granges) {
  nms <- names(granges)
  if (!is.null(nms) && !any(is.na(nms)) && !any(nms == "")) {
    return(nms)
  }
  paste0(
    as.character(GenomeInfoDb::seqnames(granges)), ":",
    BiocGenerics::start(granges), "-",
    BiocGenerics::end(granges), ":",
    as.character(BiocGenerics::strand(granges))
  )
}

# Resolve every GRanges column in parts_list to a DNAStringSet using getSeq().
# Non-GRanges columns pass through unchanged.
.resolve_granges_columns <- function(parts_list, seq_containers) {
  if (length(seq_containers) == 0L) {
    return(parts_list)
  }

  container_genomes <- vapply(seq_containers, .container_genome, character(1))

  lapply(parts_list, function(col) {
    if (!inherits(col, "GRanges")) {
      return(col)
    }

    col_genome <- unique(GenomeInfoDb::genome(GenomeInfoDb::seqinfo(col)))
    if (length(col_genome) != 1L) {
      stop("GRanges column spans multiple genome assemblies; split into one GRanges per assembly.")
    }

    idx <- match(col_genome, container_genomes)
    if (is.na(idx)) {
      stop(sprintf(
        "No seq_container found for genome \"%s\". Available: %s",
        col_genome, paste(container_genomes, collapse = ", ")
      ))
    }

    seqs <- Biostrings::getSeq(seq_containers[[idx]], col)
    names(seqs) <- .granges_names(col)
    seqs
  })
}

#' Construct a HashId
#'
#' Wraps a raw hash string as a typed \code{gen_hash_id} object used to
#' identify sequence graphs and nodes within the Gen database.
#'
#' @param hash_id Character. The hex hash string.
#' @return A \code{gen_hash_id} list.
#' @export
HashId <- function(hash_id) {
  structure(list(hash_id = as.character(hash_id)), class = "gen_hash_id")
}

#' @export
print.gen_node <- function(x, ...) {
  len <- x$sequence_end - x$sequence_start
  cat(sprintf("<Node [%d bp]>\n", len))
  invisible(x)
}

#' @export
print.gen_node_slice <- function(x, ...) {
  cat(sprintf("<NodeSlice [%d..%d] %s>\n", x$start, x$end, x$strand))
  invisible(x)
}

#' @export
print.gen_position <- function(x, ...) {
  cat(sprintf("<Position offset=%d %s>\n", x$offset, x$strand))
  invisible(x)
}

#' @export
print.gen_locus <- function(x, ...) {
  cat(sprintf("<Locus %d slice(s) %s>\n", length(x$slices), x$strand))
  invisible(x)
}

#' Create a graph plot controller
#'
#' Returns a \code{gen_plot} environment that renders the sequence graph for a
#' sequence graph.  Printing or displaying the object in RStudio / Shiny shows an
#' interactive htmlwidget.
#'
#' @param db_path Character. Path to the Gen graph SQLite database.
#' @param sequence_graph_id Character or \code{gen_hash_id}. Sequence graph to render.
#' @param detail Character. Level of node detail: \code{"normal"} (default) or
#'   \code{"compressed"}.
#' @param rows Integer or \code{NULL}. Canvas height in terminal rows (default 24).
#' @param cols Integer or \code{NULL}. Canvas width in terminal columns (default 72).
#' @return A \code{gen_plot} environment with methods:
#'   \describe{
#'     \item{\code{zoom_in()}}{Step one zoom level in. Returns self invisibly.}
#'     \item{\code{zoom_out()}}{Step one zoom level out. Returns self invisibly.}
#'     \item{\code{move_by(dx, dy)}}{Pan the viewport by \code{dx} columns and \code{dy} rows. Returns self invisibly.}
#'     \item{\code{handle_click(col, row)}}{Send a mouse click; returns \code{TRUE} if a node was hit.}
#'     \item{\code{set_detail(detail)}}{Change node detail level (\code{"normal"}, \code{"compressed"}, or \code{"full"}). Returns self invisibly.}
#'     \item{\code{render_frame(cols, rows)}}{Render to JSON string (used internally by the widget).}
#'     \item{\code{goto_match(match_locus)}}{Center the viewport on a search result locus. Sets detail to \code{"full"}. Returns self invisibly.}
#'     \item{\code{highlight_match(match_locus, color = "yellow")}}{Highlight a search result locus on the graph. \code{color} may be any named terminal color (e.g. \code{"yellow"}, \code{"red"}, \code{"cyan"}). Returns self invisibly.}
#'     \item{\code{clear_highlights()}}{Remove all highlights. Returns self invisibly.}
#'     \item{\code{add_track_file(path, name = NULL, sample = NULL)}}{Add a GFF3 or BED file as an annotation track panel below the graph. \code{sample} is the sample whose path defines the coordinate space (default \code{"reference"}).}
#'     \item{\code{add_track_group(group)}}{Add a DB-stored annotation group as a track panel below the graph.}
#'     \item{\code{list_annotations()}}{Return a list of \code{gen_annotation} records from the database. Each record has \code{id}, \code{name}, \code{group}, \code{kind}, \code{segments}, \code{length}, and \code{locus} fields.}
#'     \item{\code{go_to(annotation)}}{Navigate to an annotation returned by \code{list_annotations()}. Sets detail to \code{"full"}. Returns self invisibly.}
#'   }
#' @export
GenPlot <- function(db_path, sequence_graph_id, detail = "normal", rows = NULL, cols = NULL) {
  ctrl <- new.env(parent = emptyenv())
  ctrl$repo <- .RepositoryClass$new(dirname(dirname(db_path)))
  ctrl$sequence_graph_id <- if (inherits(sequence_graph_id, "gen_hash_id")) sequence_graph_id$hash_id else as.character(sequence_graph_id)
  ctrl$detail <- detail
  ctrl$ops <- character()
  ctrl$track_specs <- tryCatch({
    group_names <- ctrl$repo$auto_load_annotation_groups(ctrl$sequence_graph_id)
    lapply(group_names, function(n) list(type = "group", name = n))
  }, error = function(e) list())
  ctrl$rows <- rows %||% 24L
  ctrl$cols <- cols %||% 72L

  ctrl$set_detail <- function(detail) {
    ctrl$detail <- detail
    invisible(ctrl)
  }

  ctrl$render_frame <- function(cols = ctrl$cols, rows = ctrl$rows) {
    ctrl$cols <- cols
    ctrl$rows <- rows
    ctrl$repo$render_frame(ctrl$sequence_graph_id, ctrl$detail, as.integer(cols), as.integer(rows), paste(ctrl$ops, collapse = ";"), jsonlite::toJSON(ctrl$track_specs, auto_unbox = TRUE))
  }

  ctrl$add_track_group <- function(group) {
    ctrl$track_specs <- c(ctrl$track_specs, list(list(type = "group", name = group)))
    invisible(ctrl)
  }

  ctrl$add_track_file <- function(path, name = NULL, sample = NULL) {
    spec <- list(type = "file", path = path, name = name, sample = sample)
    ctrl$track_specs <- c(ctrl$track_specs, list(spec))
    invisible(ctrl)
  }


  ctrl$zoom_in <- function() {
    ctrl$ops <- c(ctrl$ops, "zi")
    invisible(ctrl)
  }

  ctrl$zoom_out <- function() {
    ctrl$ops <- c(ctrl$ops, "zo")
    invisible(ctrl)
  }

  ctrl$handle_click <- function(col, row) {
    clicked <- ctrl$repo$handle_click(
      ctrl$sequence_graph_id,
      ctrl$detail,
      paste(ctrl$ops, collapse = ";"),
      as.integer(col),
      as.integer(row)
    )
    if (isTRUE(clicked)) {
      ctrl$ops <- c(ctrl$ops, sprintf("c,%d,%d", as.integer(col), as.integer(row)))
    }
    clicked
  }

  ctrl$move_by <- function(dx, dy) {
    ctrl$ops <- c(ctrl$ops, sprintf("m,%d,%d", as.integer(dx), as.integer(dy)))
    invisible(ctrl)
  }

  ctrl$goto_match <- function(match_locus) {
    ctrl$detail <- "full"
    pos <- match_locus$start
    node <- pos$node
    offset <- pos$offset
    node_len <- node$sequence_end - node$sequence_start
    frac_x <- if (node_len > 1L) offset / (node_len - 1L) else 0.5
    frac_x <- max(0.0, min(1.0, frac_x))
    ctrl$ops <- c(ctrl$ops, sprintf("goto,%s,%d,%d,%.6f",
      node$node_id,
      as.integer(node$sequence_start),
      as.integer(node$sequence_end),
      frac_x))
    invisible(ctrl)
  }

  ctrl$highlight_match <- function(match_locus, color = "yellow") {
    slices <- match_locus$slices
    n <- length(slices)
    start_offset <- match_locus$start$offset
    end_offset <- match_locus$end$offset
    strand_code <- switch(match_locus$strand, "+" = "f", "-" = "r", "u")
    block_parts <- paste(
      sapply(slices, function(s) {
        sprintf("%s,%d,%d", s$node$node_id, as.integer(s$node$sequence_start), as.integer(s$node$sequence_end))
      }),
      collapse = ","
    )
    ctrl$ops <- c(ctrl$ops, sprintf("hl,%s,%d,%d,%s,%d,%s",
      color,
      as.integer(start_offset),
      as.integer(end_offset),
      strand_code,
      as.integer(n),
      block_parts))
    invisible(ctrl)
  }

  ctrl$clear_highlights <- function() {
    ctrl$ops <- c(ctrl$ops, "clrhl")
    invisible(ctrl)
  }

  ctrl$list_annotations <- function() {
    ctrl$repo$list_annotations(ctrl$sequence_graph_id)
  }

  ctrl$go_to <- function(x) {
    ctrl$goto_match(if (!is.null(x$locus)) x$locus else x)
  }

  class(ctrl) <- "gen_plot"
  ctrl
}

#' @export
print.gen_plot <- function(x, ...) {
  json <- x$render_frame(x$cols, x$rows)
  w <- .genplot_widget(json)
  print(w)
  invisible(x)
}

knit_print.gen_plot <- function(x, ...) {
  json <- x$render_frame(x$cols, x$rows)
  w <- .genplot_widget(json)
  knitr::knit_print(w$.get_htmlwidget(), ...)
}

methods::setOldClass("gen_plot")
methods::setMethod("show", "gen_plot", function(object) print(object))

#' Import sequences from a Bioconductor XStringSet or named character vector
#'
#' Accepts a \code{DNAStringSet}, \code{AAStringSet}, or any named character
#' vector.  Each named entry becomes one sequence graph.  No intermediate FASTA
#' file is written to disk.
#'
#' @param sequences A named character vector, \code{DNAStringSet},
#'   \code{AAStringSet}, or any \code{XStringSet}.  Names become sequence graph
#'   names and must all be non-empty.
#' @param sample Character. Sample name (default \code{"sample"}).
#' @param collection Character or \code{NULL}. Collection name; if
#'   \code{NULL} the workspace default is used.
#' @param repo A \code{gen_repository} from \code{Repository()}, or \code{NULL}
#'   to use the current working directory.
#' @return Invisible \code{NULL}.
#'
#' @examples
#' \dontrun{
#' library(Biostrings)
#' seqs <- DNAStringSet(c(chr1 = "ACGTACGT", chr2 = "TTGCTTGC"))
#' import_bioconductor(seqs, sample = "hg38")
#'
#' # Named character vector works without Biostrings
#' import_bioconductor(c(plasmid = "ATCGATCG"), sample = "ecoli")
#' }
#' @export
import_bioconductor <- function(sequences, sample = "sample", collection = NULL, repo = NULL) {
  if (!is.character(sequences)) {
    if (!requireNamespace("Biostrings", quietly = TRUE)) {
      stop(
        "Biostrings is required to use XStringSet objects. ",
        "Install it with: BiocManager::install(\"Biostrings\"). ",
        "Alternatively, pass a named character vector.",
        call. = FALSE
      )
    }
    sequences <- as.character(sequences)
  }
  nms <- names(sequences)
  if (is.null(nms) || any(is.na(nms)) || any(nms == "")) {
    stop("All sequences must have non-empty names.", call. = FALSE)
  }
  repo <- repo %||% Repository(NULL)
  repo$import_sequences(nms, unname(sequences), sample, collection)
  invisible(NULL)
}

#' Import genomic regions from a Bioconductor GRanges or data frame
#'
#' Each reference sequence is stored once as a node (identified by sequence
#' hash), so overlapping or repeated imports of the same chromosome do not
#' duplicate stored sequence data.  Each region then becomes a SequenceGraph
#' whose edges slice into that shared node at \code{[start, end)}.
#'
#' Coordinates follow Bioconductor convention on input (1-based, closed
#' \code{[start, end]}) and are converted internally to 0-based half-open
#' \code{[start, end)} for storage.
#'
#' Because all regions from the same chromosome reference the same underlying
#' node, the node hash is shared rather than duplicated — making it clear that
#' the sequence overlap is intentional and reducing storage costs when importing
#' many regions at once.
#'
#' @param regions A \code{GRanges} object (requires the \pkg{GenomicRanges}
#'   package) or a \code{data.frame} with columns \code{seqname},
#'   \code{start} (1-based inclusive), \code{end} (1-based inclusive), and
#'   \code{name}.
#' @param sequences A named character vector or \code{DNAStringSet} whose
#'   names match the \code{seqnames}/\code{seqname} values in \code{regions}.
#'   Must cover every sequence referenced by at least one region.
#' @param sample Character. Sample name (default \code{"sample"}).
#' @param collection Character or \code{NULL}. Collection name; if
#'   \code{NULL} the workspace default is used.
#' @param repo A \code{gen_repository} from \code{Repository()}, or \code{NULL}
#'   to use the current working directory.
#' @return Invisible \code{NULL}.
#'
#' @examples
#' \dontrun{
#' library(GenomicRanges)
#' library(Biostrings)
#'
#' # Full chromosome sequences
#' genome <- DNAStringSet(c(
#'   chr1 = "ACGTACGTACGTACGT",
#'   chr2 = "TTGCTTGCTTGCTTGC"
#' ))
#'
#' # Regions of interest — names become sequence graph names
#' gr <- GRanges(
#'   seqnames = c("chr1", "chr1", "chr2"),
#'   ranges   = IRanges(start = c(1, 9, 5), end = c(8, 16, 12)),
#'   names    = c("gene_a", "gene_b", "gene_c")
#' )
#' names(gr) <- c("gene_a", "gene_b", "gene_c")
#'
#' import_granges(gr, genome, sample = "hg38")
#'
#' # data.frame alternative (no GenomicRanges required)
#' regions_df <- data.frame(
#'   seqname = c("chr1", "chr1"),
#'   start   = c(1L, 9L),
#'   end     = c(8L, 16L),
#'   name    = c("gene_a", "gene_b")
#' )
#' import_granges(regions_df, genome, sample = "hg38")
#' }
#' @export
import_granges <- function(regions, sequences, sample = "sample", collection = NULL, repo = NULL) {
  if (!is.character(sequences)) {
    if (!requireNamespace("Biostrings", quietly = TRUE)) {
      stop(
        "Biostrings is required to use XStringSet objects. ",
        "Install it with: BiocManager::install(\"Biostrings\"). ",
        "Alternatively, pass a named character vector.",
        call. = FALSE
      )
    }
    sequences <- as.character(sequences)
  }
  seq_names <- names(sequences)
  if (is.null(seq_names) || any(is.na(seq_names)) || any(seq_names == "")) {
    stop("All sequences must have non-empty names.", call. = FALSE)
  }

  if (inherits(regions, "GRanges")) {
    if (!requireNamespace("GenomicRanges", quietly = TRUE)) {
      stop(
        "GenomicRanges is required to use GRanges objects. ",
        "Install it with: BiocManager::install(\"GenomicRanges\").",
        call. = FALSE
      )
    }
    region_seqnames <- as.character(GenomicRanges::seqnames(regions))
    # GRanges is 1-based closed [start, end]; convert to 0-based half-open [start, end)
    region_starts  <- as.integer(GenomicRanges::start(regions)) - 1L
    region_ends    <- as.integer(GenomicRanges::end(regions))
    region_names   <- names(regions)
    if (is.null(region_names)) {
      region_names <- paste0(region_seqnames, ":", region_starts, "-", region_ends)
    }
  } else if (is.data.frame(regions)) {
    if (!all(c("seqname", "start", "end", "name") %in% names(regions))) {
      stop("data.frame regions must have columns: seqname, start, end, name", call. = FALSE)
    }
    region_seqnames <- as.character(regions$seqname)
    region_starts   <- as.integer(regions$start) - 1L
    region_ends     <- as.integer(regions$end)
    region_names    <- as.character(regions$name)
  } else {
    stop("regions must be a GRanges or data.frame", call. = FALSE)
  }

  missing_seqs <- setdiff(region_seqnames, seq_names)
  if (length(missing_seqs) > 0L) {
    stop(
      "Regions reference sequences not found in `sequences`: ",
      paste(missing_seqs, collapse = ", "),
      call. = FALSE
    )
  }

  repo <- repo %||% Repository(NULL)
  repo$import_genomic_regions(
    seq_names, unname(sequences),
    region_names, region_seqnames,
    as.double(region_starts), as.double(region_ends),
    sample, collection
  )
  invisible(NULL)
}

#' Concatenate sequence graphs end-to-end into a new sequence graph
#'
#' @param repo A \code{Repository} object.
#' @param bgs List of \code{SequenceGraph} objects, all in the same collection
#'   and sample.
#' @param new_sample Character. Sample name for the stitched result.
#' @param new_region Character. Name for the stitched sequence graph.
#' @return A \code{SequenceGraph}.
#' @export
stitch <- function(repo, bgs, new_sample, new_region) {
  if (length(bgs) == 0L) stop("stitch() requires at least one sequence graph", call. = FALSE)
  collection  <- bgs[[1L]]$collection()
  sample_name <- bgs[[1L]]$sample_name()
  for (bg in bgs[-1L]) {
    if (bg$collection() != collection)
      stop(sprintf("All sequence graphs must be in the same collection ('%s' vs '%s')",
                   collection, bg$collection()), call. = FALSE)
    if (bg$sample_name() != sample_name)
      stop(sprintf("All sequence graphs must be in the same sample ('%s' vs '%s')",
                   sample_name, bg$sample_name()), call. = FALSE)
  }
  regions <- paste(sapply(bgs, function(bg) bg$name()), collapse = ",")
  repo$stitch(collection, sample_name, new_sample, new_region, regions)
}

#' Get the sequence string for a node
#'
#' @param obj A \code{Repository} or \code{SequenceGraph}.
#' @param node A \code{gen_node} record (from \code{to_dict()$nodes}).
#' @return Character string of the node sequence.
#' @export
get_node_sequence <- function(obj, node) {
  obj$get_node_sequence(
    node$node_id,
    as.integer(node$sequence_start),
    as.integer(node$sequence_end)
  )
}

#' @export
plot.SequenceGraph <- function(x, rows = NULL, cols = NULL, detail = "normal", ...) {
  GenPlot(x$db_path(), x$id(), detail = detail, rows = rows, cols = cols)
}

#' The sequence graphs produced by a single import/update/derive call
#'
#' A \code{gen_sample} is a read-only list of \code{SequenceGraph} objects,
#' all within one \code{collection_name}/\code{sample_name}. Index it with
#' \code{[[}, iterate it, or call \code{length()} on it.
#'
#' @param x A \code{gen_sample}.
#' @export
length.gen_sample <- function(x) length(x$block_groups)

#' @export
`[[.gen_sample` <- function(x, i) x$block_groups[[i]]

#' @export
`[.gen_sample` <- function(x, i) x$block_groups[i]

#' @export
print.gen_sample <- function(x, ...) {
  cat(sprintf(
    "<gen_sample> %s (collection=%s, %d sequence graph%s)\n",
    x$sample_name, x$collection_name, length(x),
    if (length(x) == 1L) "" else "s"
  ))
  for (i in seq_along(x$block_groups)) {
    cat(sprintf("  [[%d]]: %s\n", i, x$block_groups[[i]]$name()))
  }
  invisible(x)
}

#' Plot the first sequence graph in a sample
#'
#' @param x A \code{gen_sample}.
#' @param ... Passed to \code{plot.SequenceGraph}.
#' @export
plot.gen_sample <- function(x, ...) {
  plot(x$block_groups[[1L]], ...)
}

#' Import a combinatorial library
#'
#' Thin wrapper around \code{repo$import_library()}.
#' Each column in \code{parts_list} must be a named character vector or
#' \code{DNAStringSet}; names become annotation names.
#'
#' @param repo A \code{Repository}.
#' @param library_name Character. Library name.
#' @param parts_list Named list of part vectors (named character vector or DNAStringSet).
#' @param sample Character or \code{NULL}.
#' @param collection Character or \code{NULL}.
#' @return The imported \code{SequenceGraph}.
#' @export
import_library <- function(repo, library_name, parts_list, sample = NULL, collection = NULL) {
  repo$import_library(library_name, parts_list, sample, collection)
}

#' Apply a library update
#'
#' @param repo A \code{Repository}.
#' @param sample Character or \code{NULL}. Source sample.
#' @param new_sample_name Character. New sample name.
#' @param path_name Character. Path to update.
#' @param parts_list Named list of part vectors (named character vector or DNAStringSet).
#' @param collection Character or \code{NULL}.
#' @return The updated \code{gen_sample}.
#' @export
update_with_library <- function(repo, sample = NULL, new_sample_name, path_name,
                                parts_list, collection = NULL) {
  repo$update_with_library(sample, new_sample_name, path_name, parts_list, collection)
}

#' Import a combinatorial library from GRanges part definitions
#'
#' Like \code{import_library()}, but each column in \code{parts_list} may be a
#' \code{GRanges} object.  Sequences are extracted via \code{Biostrings::getSeq()}
#' using the matching \code{seq_containers} entry (matched by genome assembly name
#' in the \code{Seqinfo} slot).  Named character vector and \code{DNAStringSet}
#' columns pass through unchanged.
#'
#' If a \code{GRanges} column has no \code{names()}, part names are
#' auto-generated from coordinates as \code{"<seqname>:<start>-<end>:<strand>"}.
#'
#' @param repo A \code{Repository}.
#' @param library_name Character. Library name.
#' @param parts_list List of part columns (named character vector, DNAStringSet, or GRanges).
#' @param seq_containers List of sequence containers (BSgenome, FaFile, TwoBitFile, or
#'   DNAStringSet with \code{metadata(x)$genome} set) used to resolve GRanges columns.
#' @param sample Character or \code{NULL}.
#' @param collection Character or \code{NULL}.
#' @return The imported \code{SequenceGraph}.
#' @export
import_library_from_granges <- function(repo, library_name, parts_list,
                                        seq_containers = list(),
                                        sample = NULL, collection = NULL) {
  parts_list <- .resolve_granges_columns(parts_list, seq_containers)
  repo$import_library(library_name, parts_list, sample, collection)
}

#' Apply a library update from GRanges part definitions
#'
#' Like \code{update_with_library()}, but each column in \code{parts_list} may
#' be a \code{GRanges} object.  See \code{\link{import_library_from_granges}}
#' for details on \code{seq_containers} and automatic name generation.
#'
#' @param repo A \code{Repository}.
#' @param sample Character or \code{NULL}. Source sample.
#' @param new_sample_name Character. New sample name.
#' @param path_name Character. Path to update.
#' @param parts_list List of part columns (named character vector, DNAStringSet, or GRanges).
#' @param seq_containers List of sequence containers for GRanges resolution.
#' @param collection Character or \code{NULL}.
#' @return The updated \code{gen_sample}.
#' @export
update_with_library_from_granges <- function(repo, sample = NULL, new_sample_name,
                                             path_name, parts_list,
                                             seq_containers = list(),
                                             collection = NULL) {
  parts_list <- .resolve_granges_columns(parts_list, seq_containers)
  repo$update_with_library(sample, new_sample_name, path_name, parts_list, collection)
}

# Save the extendr Repository class before Repository() overwrites the name.
.RepositoryClass <- Repository

# Patch the generated $.Repository so it resolves against .RepositoryClass
# rather than the name "Repository", which is overwritten below.
`$.Repository` <- function(self, name) {
  func <- .RepositoryClass[[name]]
  environment(func) <- environment()
  func
}
`[[.Repository` <- `$.Repository`

#' Open a Gen repository
#'
#' @param path Character or \code{NULL}. Workspace root; defaults to the current
#'   working directory.
#' @return A \code{gen_repository} environment.
#' @export
Repository <- function(path = NULL) {
  inner <- .RepositoryClass$new(path)
  repo  <- new.env(parent = emptyenv())
  class(repo) <- "gen_repository"
  repo$.inner  <- inner
  repo$gen_dir <- inner$gen_dir()
  repo$db_path <- inner$db_path()

  repo$execute    <- function(query) inner$execute(query)
  repo$query      <- function(query) inner$query(query)
  repo$get_sequence_graph_by_id          <- function(id) inner$get_sequence_graph_by_id(as.character(id))
  repo$get_sequence_graphs               <- function() inner$get_sequence_graphs()
  repo$get_sequence_graphs_by_collection <- function(collection) inner$get_sequence_graphs_by_collection(collection)
  repo$build_index    <- function(ids = character(), sequence_kind = "dna", k = 16L) inner$build_index(ids, sequence_kind, as.integer(k))
  repo$search         <- function(query, ids = character(), sequence_kind = "dna") inner$search(query, ids, sequence_kind)
  repo$clear_index    <- function(ids = character()) inner$clear_index(ids)
  repo$render_frame   <- function(sg_id, detail, cols, rows, ops, tracks_json) inner$render_frame(sg_id, detail, cols, rows, ops, tracks_json)
  repo$handle_click   <- function(sg_id, detail, ops, col, row) inner$handle_click(sg_id, detail, ops, col, row)
  repo$list_annotations           <- function(sg_id) inner$list_annotations(sg_id)
  repo$auto_load_annotation_groups <- function(sg_id) inner$auto_load_annotation_groups(sg_id)
  repo$get_node_sequence          <- function(node_id, sequence_start, sequence_end) inner$get_node_sequence(node_id, as.integer(sequence_start), as.integer(sequence_end))

  repo$import_fasta           <- function(filename, sample = "sample", shallow = FALSE, collection = NULL) inner$import_fasta(filename, sample, isTRUE(shallow), collection)
  repo$import_reference_fasta <- function(filename, reference, shallow = FALSE, collection = NULL) inner$import_reference_fasta(filename, reference, isTRUE(shallow), collection)
  repo$import_gfa             <- function(filename, sample = "sample", collection = NULL) inner$import_gfa(filename, sample, collection)
  repo$import_genbank         <- function(filename, sample = "sample", collection = NULL) inner$import_genbank(filename, sample, collection)
  repo$import_library_files   <- function(library_name, parts, library, sample = "sample", collection = NULL) inner$import_library_files(library_name, parts, library, sample, collection)
  repo$import_sequences       <- function(names, sequences, sample = "sample", collection = NULL) inner$import_sequences(names, sequences, sample, collection)
  repo$import_genomic_regions <- function(seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample = "sample", collection = NULL) inner$import_genomic_regions(seq_names, seq_sequences, region_names, region_seq_names, region_starts, region_ends, sample, collection)
  repo$import_library         <- function(library_name, parts_list, sample = NULL, collection = NULL) inner$import_library(library_name, parts_list, sample, collection)

  repo$update_with_fasta          <- function(filename, sample, new_sample, region_name, collection = NULL) inner$update_with_fasta(filename, sample, new_sample, region_name, collection)
  repo$update_with_gfa            <- function(filename, sample, new_sample, collection = NULL) inner$update_with_gfa(filename, sample, new_sample, collection)
  repo$update_with_gaf            <- function(filename, csv, sample, parent_sample = NULL, collection = NULL) inner$update_with_gaf(filename, csv, sample, parent_sample, collection)
  repo$update_with_vcf            <- function(filename, genotype = NULL, sample = NULL, reference = character(), in_place = FALSE, collection = NULL) inner$update_with_vcf(filename, genotype, sample, reference, isTRUE(in_place), collection)
  repo$update_with_genbank        <- function(filename, sample, create_missing = FALSE, collection = NULL) inner$update_with_genbank(filename, sample, isTRUE(create_missing), collection)
  repo$update_with_sequence       <- function(sequence, sample, new_sample, region_name, no_reference_path_update = FALSE, collection = NULL) inner$update_with_sequence(sequence, sample, new_sample, region_name, isTRUE(no_reference_path_update), collection)
  repo$update_with_library_files  <- function(sample, new_sample, path_name, library, parts, collection = NULL) inner$update_with_library_files(sample, new_sample, path_name, library, parts, collection)
  repo$update_with_library        <- function(sample = NULL, new_sample_name, path_name, parts_list, collection = NULL) inner$update_with_library(sample, new_sample_name, path_name, parts_list, collection)

  repo$export_fasta   <- function(filename, sample = NULL, collection = NULL) inner$export_fasta(filename, sample, collection)
  repo$export_gfa     <- function(filename, sample, node_max = NULL, collection = NULL) inner$export_gfa(filename, sample, node_max, collection)
  repo$export_genbank <- function(filename, sample, collection = NULL) inner$export_genbank(filename, sample, collection)

  repo$derive_subgraph <- function(sample, new_sample, region, backbone = NULL, collection = NULL) inner$derive_subgraph(collection, sample, new_sample, region, backbone)
  repo$derive_chunks   <- function(sample, new_sample, region, backbone = NULL, breakpoints = NULL, chunk_size = NULL, collection = NULL)
    inner$derive_chunks(collection, sample, new_sample, region, backbone,
                        if (is.null(breakpoints)) integer(0) else as.integer(breakpoints),
                        if (is.null(chunk_size)) NULL else as.integer(chunk_size))

  repo$stitch <- function(collection_name, sample_name, new_sample, new_region, regions)
    inner$stitch(collection_name, sample_name, new_sample, new_region, regions)

  repo
}
