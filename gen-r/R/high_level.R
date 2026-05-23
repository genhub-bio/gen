#' @useDynLib genr, .registration = TRUE
NULL

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Construct a HashId
#'
#' Wraps a raw hash string as a typed \code{gen_hash_id} object used to
#' identify block groups and nodes within the Gen database.
#'
#' @param hash_id Character. The hex hash string.
#' @return A \code{gen_hash_id} list.
#' @export
HashId <- function(hash_id) {
  structure(list(hash_id = as.character(hash_id)), class = "gen_hash_id")
}

#' Construct a Block
#'
#' A block identifies a contiguous byte range within a graph node.
#' Pass the result to \code{repo$get_block_sequence()} to retrieve the
#' underlying sequence.
#'
#' @param node_id Character or \code{gen_hash_id}. Node identifier.
#' @param sequence_start Integer. Start byte offset (inclusive).
#' @param sequence_end Integer. End byte offset (exclusive).
#' @return A \code{gen_block} list.
#' @export
Block <- function(node_id, sequence_start, sequence_end) {
  structure(
    list(
      node_id = if (inherits(node_id, "gen_hash_id")) node_id else HashId(node_id),
      sequence_start = as.integer(sequence_start),
      sequence_end = as.integer(sequence_end)
    ),
    class = "gen_block"
  )
}


# Extract the genome string from a sequence container.
# Supported: BSgenome, DNAStringSet (with metadata$genome set), FaFile, TwoBitFile.
.container_genome <- function(container) {
  if (inherits(container, "BSgenome")) {
    return(unique(GenomeInfoDb::genome(GenomeInfoDb::seqinfo(container))))
  }
  # DNAStringSet, FaFile, TwoBitFile: genome must be set via metadata(x)$genome.
  meta <- S4Vectors::metadata(container)
  meta[["genome"]] %||% stop(
    "seq_container has no genome metadata. Set it with: metadata(x)$genome <- \"<assembly>\""
  )
}

# Resolve every GRanges column in parts_list to a DNAStringSet using getSeq().
# Non-GRanges columns pass through unchanged.
#
# Future: replace getSeq() with coordinate-based part references so sequences
# already in the gen database are reused rather than copied as flat strings.
# The flow would be:
#   1. Match each unique seqname to an existing Sequence in the DB by name.
#   2. Auto-import missing contigs from the matched container (getSeq one contig
#      at a time, import_fasta into DB, retrieve sequence_hash).
#   3. Construct (sequence_hash, start, end, strand, part_name) tuples and pass
#      a new Rust entry point that creates coordinate-based Node slices instead
#      of copying strings into SequencePart records.
resolve_granges_columns <- function(parts_list, seq_containers) {
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

    Biostrings::getSeq(seq_containers[[idx]], col)
  })
}

#' Construct a DbContext
#'
#' Opens the Gen graph database and operations database, returning a context
#' object used by low-level Rust entry points.  Prefer \code{Repository()}
#' for day-to-day use.
#'
#' @param workspace_path Character or \code{NULL}. Path to the workspace root.
#'   Defaults to the current working directory.
#' @param db_path Character or \code{NULL}. Explicit path to the graph SQLite
#'   database.  When \code{NULL} the path is derived from the workspace.
#' @return A \code{gen_db_context} object.
#' @export
DbContext <- function(workspace_path = NULL, db_path = NULL) {
  structure(db_context(workspace_path = workspace_path, db_path = db_path), class = "gen_db_context")
}

.as_block_group <- function(x, gen_dir = NULL) {
  bg <- new.env(parent = emptyenv())
  bg$id <- HashId(x$id)
  bg$collection_name <- x$collection_name
  bg$sample_name <- x$sample_name
  bg$name <- x$name
  bg$db_path <- x$db_path %||% NULL
  bg$gen_dir <- gen_dir

  bg$plot <- function(rows = NULL, cols = NULL, detail = "normal") {
    if (is.null(bg$db_path)) {
      stop("plot() requires a db_path; obtain BlockGroup via Repository()", call. = FALSE)
    }
    GenPlot(bg$db_path, bg$id$hash_id, detail = detail, rows = rows, cols = cols)
  }

  bg$export_fasta <- function(filename) {
    if (is.null(bg$db_path)) stop("export_fasta() requires a db_path", call. = FALSE)
    repo_bg_export_fasta(bg$db_path, bg$collection_name, bg$sample_name, filename)
  }

  bg$export_gfa <- function(filename, node_max = NULL) {
    if (is.null(bg$db_path)) stop("export_gfa() requires a db_path", call. = FALSE)
    repo_bg_export_gfa(bg$db_path, bg$collection_name, bg$sample_name, filename, node_max)
  }

  bg$export_genbank <- function(filename) {
    if (is.null(bg$db_path)) stop("export_genbank() requires a db_path", call. = FALSE)
    repo_bg_export_genbank(bg$db_path, bg$collection_name, bg$sample_name, filename)
  }

  bg$build_index <- function(sequence_kind = "dna", k = 16L) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("build_index() requires a Repository context", call. = FALSE)
    repo_build_index(bg$db_path, bg$gen_dir, c(bg$id$hash_id), sequence_kind, as.integer(k))
  }

  bg$search <- function(query, sequence_kind = "dna") {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("search() requires a Repository context", call. = FALSE)
    result <- repo_search(bg$db_path, bg$gen_dir, query, c(bg$id$hash_id), sequence_kind)
    if (length(result) == 0L) list() else result[[1L]]$matches
  }

  bg$clear_index <- function() {
    if (is.null(bg$gen_dir)) stop("clear_index() requires a Repository context", call. = FALSE)
    repo_clear_index(bg$gen_dir, c(bg$id$hash_id))
  }

  bg$subgraph <- function(new_sample, start, end, backbone = NULL) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("subgraph() requires a Repository context", call. = FALSE)
    workspace_path <- dirname(bg$gen_dir)
    result <- repo_bg_subgraph(
      workspace_path, bg$db_path, bg$collection_name, bg$sample_name,
      bg$name, new_sample, as.integer(start), as.integer(end), backbone
    )
    .as_block_group(result, gen_dir = bg$gen_dir)
  }

  bg$chunks <- function(new_sample, breakpoints = NULL, chunk_size = NULL, backbone = NULL) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("chunks() requires a Repository context", call. = FALSE)
    workspace_path <- dirname(bg$gen_dir)
    results <- repo_bg_chunks(
      workspace_path, bg$db_path, bg$collection_name, bg$sample_name,
      bg$name, new_sample, breakpoints,
      if (is.null(chunk_size)) NULL else as.integer(chunk_size),
      backbone
    )
    lapply(results, .as_block_group, gen_dir = bg$gen_dir)
  }

  class(bg) <- "gen_block_group"
  bg
}

#' Create a graph plot controller
#'
#' Returns a \code{gen_plot} environment that renders the sequence graph for a
#' block group.  Printing or displaying the object in RStudio / Shiny shows an
#' interactive htmlwidget.
#'
#' @param db_path Character. Path to the Gen graph SQLite database.
#' @param block_group_id Character or \code{gen_hash_id}. Block group to render.
#' @param detail Character. Level of node detail: \code{"normal"} (default) or
#'   \code{"compressed"}.
#' @param rows Integer or \code{NULL}. Canvas height in terminal rows (default 24).
#' @param cols Integer or \code{NULL}. Canvas width in terminal columns (default 80).
#' @return A \code{gen_plot} environment with methods:
#'   \describe{
#'     \item{\code{zoom_in()}}{Step one zoom level in. Returns self invisibly.}
#'     \item{\code{zoom_out()}}{Step one zoom level out. Returns self invisibly.}
#'     \item{\code{move_by(dx, dy)}}{Pan the viewport by \code{dx} columns and \code{dy} rows. Returns self invisibly.}
#'     \item{\code{handle_click(col, row)}}{Send a mouse click; returns \code{TRUE} if a node was hit.}
#'     \item{\code{set_detail(detail)}}{Change node detail level. Returns self invisibly.}
#'     \item{\code{render_frame(cols, rows)}}{Render to JSON string (used internally by the widget).}
#'   }
#' @export
GenPlot <- function(db_path, block_group_id, detail = "normal", rows = NULL, cols = NULL) {
  ctrl <- new.env(parent = emptyenv())
  ctrl$db_path <- db_path
  ctrl$block_group_id <- if (inherits(block_group_id, "gen_hash_id")) block_group_id$hash_id else as.character(block_group_id)
  ctrl$detail <- detail
  ctrl$ops <- character()
  ctrl$track_specs <- list()
  ctrl$rows <- rows %||% 24L
  ctrl$cols <- cols %||% 80L

  ctrl$set_detail <- function(detail) {
    ctrl$detail <- detail
    invisible(ctrl)
  }

  ctrl$render_frame <- function(cols = ctrl$cols, rows = ctrl$rows) {
    ctrl$cols <- cols
    ctrl$rows <- rows
    graph_render_frame(ctrl$db_path, ctrl$block_group_id, ctrl$detail, as.integer(cols), as.integer(rows), paste(ctrl$ops, collapse = ";"), jsonlite::toJSON(ctrl$track_specs, auto_unbox = TRUE))
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

  ctrl$clear_tracks <- function() {
    ctrl$track_specs <- list()
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
    clicked <- graph_handle_click(
      ctrl$db_path,
      ctrl$block_group_id,
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
    block <- match_locus$start$block
    offset <- match_locus$start$offset
    node_len <- block$sequence_end - block$sequence_start
    frac_x <- if (node_len > 1L) offset / (node_len - 1L) else 0.5
    frac_x <- max(0.0, min(1.0, frac_x))
    ctrl$ops <- c(ctrl$ops, sprintf("goto,%s,%d,%d,%.6f",
      block$node_id,
      as.integer(block$sequence_start),
      as.integer(block$sequence_end),
      frac_x))
    invisible(ctrl)
  }

  ctrl$highlight_match <- function(match_locus, color = "yellow") {
    blocks <- match_locus$blocks
    n <- length(blocks)
    start_offset <- match_locus$start$offset
    end_offset <- match_locus$end$offset
    strand_code <- switch(match_locus$strand, forward = "f", reverse = "r", "u")
    block_parts <- paste(
      sapply(blocks, function(b) {
        sprintf("%s,%d,%d", b$node_id, as.integer(b$sequence_start), as.integer(b$sequence_end))
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

methods::setOldClass("gen_plot")
methods::setMethod("show", "gen_plot", function(object) print(object))

#' Import sequences from a Bioconductor XStringSet or named character vector
#'
#' Accepts a \code{DNAStringSet}, \code{AAStringSet}, or any named character
#' vector.  Each named entry becomes one block group.  No intermediate FASTA
#' file is written to disk.
#'
#' @param sequences A named character vector, \code{DNAStringSet},
#'   \code{AAStringSet}, or any \code{XStringSet}.  Names become block group
#'   names and must all be non-empty.
#' @param sample Character. Sample name (default \code{"sample"}).
#' @param collection_name Character or \code{NULL}. Collection name; if
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
import_bioconductor <- function(sequences, sample = "sample", collection_name = NULL, repo = NULL) {
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
  workspace_path <- if (is.null(repo)) NULL else dirname(repo$gen_dir)
  db_path        <- if (is.null(repo)) NULL else repo$db_path
  import_sequences(
    workspace_path  = workspace_path,
    db_path         = db_path,
    names           = nms,
    sequences       = unname(sequences),
    sample          = sample,
    collection_name = collection_name
  )
  invisible(NULL)
}

#' Import genomic regions from a Bioconductor GRanges or data frame
#'
#' Each reference sequence is stored once as a node (identified by sequence
#' hash), so overlapping or repeated imports of the same chromosome do not
#' duplicate stored sequence data.  Each region then becomes a BlockGroup
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
#' @param collection_name Character or \code{NULL}. Collection name; if
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
#' # Regions of interest — names become block group names
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
import_granges <- function(regions, sequences, sample = "sample", collection_name = NULL, repo = NULL) {
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

  workspace_path <- if (is.null(repo)) NULL else dirname(repo$gen_dir)
  db_path        <- if (is.null(repo)) NULL else repo$db_path
  import_genomic_regions(
    workspace_path   = workspace_path,
    db_path          = db_path,
    seq_names        = seq_names,
    seq_sequences    = unname(sequences),
    region_names     = region_names,
    region_seq_names = region_seqnames,
    region_starts    = region_starts,
    region_ends      = region_ends,
    sample           = sample,
    collection_name  = collection_name
  )
  invisible(NULL)
}

#' Open a Gen repository
#'
#' The main entry point for working with a Gen sequence graph database.
#' Discovers the \code{.gen/} directory from \code{path} (or the current
#' working directory) and opens both the graph and operations databases.
#'
#' @param path Character or \code{NULL}. Path to the workspace root.  When
#'   \code{NULL} the current working directory is used.
#' @return A \code{gen_repository} environment with the following methods:
#'   \describe{
#'     \item{\code{import_fasta(filename, sample="sample", shallow=FALSE, collection_name=NULL)}}{Import a FASTA file as a new block group.}
#'     \item{\code{import_gfa(filename, sample="sample", collection_name=NULL)}}{Import a GFA file.}
#'     \item{\code{import_genbank(filename, sample="sample", collection_name=NULL)}}{Import a GenBank file (plain or gzipped).}
#'     \item{\code{import_library(library_name, parts_list, sample=NULL, collection_name=NULL)}}{Import a combinatorial sequence library.}
#'     \item{\code{import_library_files(library_name, parts, library, sample="sample", collection_name=NULL)}}{Import a library from parts/library CSV files.}
#'     \item{\code{export_fasta(filename, sample=NULL, collection_name=NULL)}}{Export sequences to FASTA.}
#'     \item{\code{export_gfa(filename, sample, node_max=NULL, collection_name=NULL)}}{Export to GFA.}
#'     \item{\code{export_genbank(filename, sample, collection_name=NULL)}}{Export to GenBank.}
#'     \item{\code{update_with_fasta(filename, sample, new_sample, region_name, start, end, collection_name=NULL)}}{Apply a FASTA update to an existing block group.}
#'     \item{\code{update_with_gfa(filename, sample, new_sample, collection_name=NULL)}}{Apply a GFA update.}
#'     \item{\code{update_with_gaf(filename, csv, sample, parent_sample=NULL, collection_name=NULL)}}{Apply a GAF update.}
#'     \item{\code{update_with_vcf(filename, genotype=NULL, sample=NULL, parent_samples=character(), in_place=FALSE, collection_name=NULL)}}{Apply a VCF update.}
#'     \item{\code{update_with_genbank(filename, sample, create_missing=FALSE, collection_name=NULL)}}{Apply a GenBank update.}
#'     \item{\code{update_with_sequence(sequence, sample, new_sample, region_name, start, end, no_reference_path_update=FALSE, collection_name=NULL)}}{Apply a raw sequence update.}
#'     \item{\code{update_with_library(sample=NULL, new_sample_name, path_name, start, end, parts_list, collection_name=NULL)}}{Apply a library update.}
#'     \item{\code{update_with_library_files(sample, new_sample, path_name, start, end, library, parts, collection_name=NULL)}}{Apply a library-files update.}
#'     \item{\code{get_block_groups()}}{Return a list of all block groups.}
#'     \item{\code{get_block_group_by_id(id)}}{Return a block group by its \code{HashId}.}
#'     \item{\code{get_block_groups_by_collection(collection_name)}}{Return block groups in a collection.}
#'     \item{\code{get_block_sequence(block)}}{Return the sequence string for a \code{Block}.}
#'     \item{\code{plot(block_group, rows, cols, detail)}}{Return a \code{gen_plot} for a block group.}
#'     \item{\code{stitch(bgs, new_sample, new_region)}}{Concatenate block groups end-to-end into a new block group.}
#'     \item{\code{make_stitch(sample, new_sample, regions, new_region, collection_name=NULL)}}{Stitch named regions into a new block group by name.}
#'     \item{\code{derive_subgraph(sample, new_sample, region, backbone=NULL, collection_name=NULL)}}{Derive a subgraph block group.}
#'     \item{\code{derive_chunks(sample, new_sample, region, backbone=NULL, breakpoints=NULL, chunk_size=NULL, collection_name=NULL)}}{Split a block group into chunks.}
#'     \item{\code{build_index(bgs, sequence_kind, k)}}{Build a k-mer seed index to accelerate \code{search()}.}
#'     \item{\code{search(query, bgs, sequence_kind)}}{Search for exact sequence occurrences across block groups.}
#'     \item{\code{clear_index(bgs)}}{Remove cached search index files.}
#'     \item{\code{execute(query)}}{Run a raw SQL statement against the graph database.}
#'     \item{\code{query(query)}}{Run a raw SQL query and return results as a list of rows.}
#'   }
#' @export
Repository <- function(path = NULL) {
  repo <- new.env(parent = emptyenv())
  repo$gen_dir <- repo_get_gen_dir(path)
  repo$db_path <- repo_get_db_path(path)

  repo$execute <- function(query) {
    repo_execute(repo$db_path, query)
  }

  repo$query <- function(query) {
    repo_query(repo$db_path, query)
  }

  repo$get_block_group_by_id <- function(id) {
    id_str <- if (inherits(id, "gen_hash_id")) id$hash_id else as.character(id)
    .as_block_group(repo_get_block_group_by_id(repo$db_path, id_str), gen_dir = repo$gen_dir)
  }

  repo$get_block_groups <- function() {
    lapply(repo_get_block_groups(repo$db_path), .as_block_group, gen_dir = repo$gen_dir)
  }

  repo$get_block_groups_by_collection <- function(collection_name) {
    lapply(
      repo_get_block_groups_by_collection(repo$db_path, collection_name),
      .as_block_group,
      gen_dir = repo$gen_dir
    )
  }

  repo$block_group_to_dict <- function(block_group) {
    repo_block_group_to_dict(repo$db_path, block_group$id$hash_id)
  }

  repo$block_group_to_rustworkx <- function(block_group) {
    repo$block_group_to_dict(block_group)
  }

  repo$block_group_to_networkx <- function(block_group) {
    repo$block_group_to_dict(block_group)
  }

  repo$plot <- function(block_group, rows = NULL, cols = NULL, detail = "normal") {
    GenPlot(repo$db_path, block_group$id$hash_id, detail = detail, rows = rows, cols = cols)
  }

  repo$get_block_sequence <- function(block) {
    repo_get_block_sequence(
      repo$db_path,
      block$node_id$hash_id,
      as.integer(block$sequence_start),
      as.integer(block$sequence_end)
    )
  }

  repo$stitch <- function(bgs, new_sample, new_region) {
    if (length(bgs) == 0L) stop("stitch() requires at least one block group", call. = FALSE)
    collection_name <- bgs[[1L]]$collection_name
    sample_name <- bgs[[1L]]$sample_name
    for (bg in bgs[-1L]) {
      if (bg$collection_name != collection_name)
        stop(sprintf("All block groups must be in the same collection ('%s' vs '%s')",
                     collection_name, bg$collection_name), call. = FALSE)
      if (bg$sample_name != sample_name)
        stop(sprintf("All block groups must be in the same sample ('%s' vs '%s')",
                     sample_name, bg$sample_name), call. = FALSE)
    }
    regions <- paste(sapply(bgs, function(bg) bg$name), collapse = ",")
    workspace_path <- dirname(repo$gen_dir)
    result <- repo_stitch(
      workspace_path, repo$db_path, collection_name, sample_name,
      new_sample, new_region, regions
    )
    .as_block_group(result, gen_dir = repo$gen_dir)
  }

  repo$build_index <- function(bgs = NULL, sequence_kind = "dna", k = 16L) {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    repo_build_index(repo$db_path, repo$gen_dir, ids, sequence_kind, as.integer(k))
  }

  repo$search <- function(query, bgs = NULL, sequence_kind = "dna") {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    results <- repo_search(repo$db_path, repo$gen_dir, query, ids, sequence_kind)
    lapply(results, function(r) {
      list(
        block_group = .as_block_group(r$block_group, gen_dir = repo$gen_dir),
        matches = r$matches
      )
    })
  }

  repo$clear_index <- function(bgs = NULL) {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    repo_clear_index(repo$gen_dir, ids)
  }

  repo$import_fasta <- function(filename, sample = "sample", shallow = FALSE, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    import_fasta(workspace_path, repo$db_path, filename, sample, shallow, collection_name)
  }

  repo$import_gfa <- function(filename, sample = "sample", collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    import_gfa(workspace_path, repo$db_path, filename, sample, collection_name)
  }

  repo$import_genbank <- function(filename, sample = "sample", collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    import_genbank(workspace_path, repo$db_path, filename, sample, collection_name)
  }

  repo$import_library <- function(library_name, parts_list, seq_containers = list(), sample = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    parts_list <- resolve_granges_columns(parts_list, seq_containers)
    import_library(workspace_path, repo$db_path, library_name, parts_list, sample, collection_name)
  }

  repo$import_library_files <- function(library_name, parts, library, sample = "sample", collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    import_library_files(workspace_path, repo$db_path, library_name, parts, library, sample, collection_name)
  }

  repo$update_with_fasta <- function(filename, sample, new_sample, region_name, start, end, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_fasta(workspace_path, repo$db_path, filename, sample, new_sample, region_name, as.integer(start), as.integer(end), collection_name)
  }

  repo$update_with_gfa <- function(filename, sample, new_sample, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_gfa(workspace_path, repo$db_path, filename, sample, new_sample, collection_name)
  }

  repo$update_with_gaf <- function(filename, csv, sample, parent_sample = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_gaf(workspace_path, repo$db_path, filename, csv, sample, parent_sample, collection_name)
  }

  repo$update_with_vcf <- function(filename, genotype = NULL, sample = NULL, parent_samples = character(), in_place = FALSE, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_vcf(workspace_path, repo$db_path, filename, genotype, sample, parent_samples, in_place, collection_name)
  }

  repo$update_with_genbank <- function(filename, sample, create_missing = FALSE, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_genbank(workspace_path, repo$db_path, filename, sample, create_missing, collection_name)
  }

  repo$update_with_sequence <- function(sequence, sample, new_sample, region_name, start, end, no_reference_path_update = FALSE, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_sequence(workspace_path, repo$db_path, sequence, sample, new_sample, region_name, as.integer(start), as.integer(end), no_reference_path_update, collection_name)
  }

  repo$update_with_library <- function(sample = NULL, new_sample_name, path_name, start, end, parts_list, seq_containers = list(), collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    parts_list <- resolve_granges_columns(parts_list, seq_containers)
    update_with_library(workspace_path, repo$db_path, sample, new_sample_name, path_name, as.integer(start), as.integer(end), parts_list, collection_name)
  }

  repo$update_with_library_files <- function(sample, new_sample, path_name, start, end, library, parts, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    update_with_library_files(workspace_path, repo$db_path, sample, new_sample, path_name, as.integer(start), as.integer(end), library, parts, collection_name)
  }

  repo$export_fasta <- function(filename, sample = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    export_fasta(workspace_path, repo$db_path, filename, sample, collection_name)
  }

  repo$export_gfa <- function(filename, sample, node_max = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    export_gfa(workspace_path, repo$db_path, filename, sample, node_max, collection_name)
  }

  repo$export_genbank <- function(filename, sample, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    export_genbank(workspace_path, repo$db_path, filename, sample, collection_name)
  }

  repo$derive_subgraph <- function(sample, new_sample, region, backbone = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    derive_subgraph(workspace_path, repo$db_path, sample, new_sample, region, backbone, collection_name)
  }

  repo$derive_chunks <- function(sample, new_sample, region, backbone = NULL, breakpoints = NULL, chunk_size = NULL, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    derive_chunks(workspace_path, repo$db_path, sample, new_sample, region, backbone, breakpoints, chunk_size, collection_name)
  }

  repo$make_stitch <- function(sample, new_sample, regions, new_region, collection_name = NULL) {
    workspace_path <- dirname(repo$gen_dir)
    make_stitch(workspace_path, repo$db_path, sample, new_sample, regions, new_region, collection_name)
  }

  class(repo) <- "gen_repository"
  repo
}
