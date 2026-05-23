\name{HashId}
\alias{HashId}
\alias{Block}
\alias{DbContext}
\alias{GraphController}
\alias{Repository}
\title{Constructors for Gen R Objects}
\usage{
HashId(hash_id)

Block(node_id, sequence_start, sequence_end)

DbContext(workspace_path = NULL, db_path = NULL)

GraphController(
  db_path,
  block_group_id,
  detail = "normal",
  rows = NULL,
  cols = NULL
)

Repository(path = NULL)
}
\arguments{
  \item{hash_id}{Hash identifier string.}

  \item{node_id}{Node identifier string or a \code{gen_hash_id} object.}

  \item{sequence_start}{Start offset within a node sequence.}

  \item{sequence_end}{End offset within a node sequence.}

  \item{workspace_path}{Optional workspace root passed through to \code{\link{db_context}}.}

  \item{db_path}{Path to a Gen SQLite database.}

  \item{block_group_id}{Block group hash ID string or \code{gen_hash_id} object.}

  \item{detail}{Graph detail level: \code{"normal"}, \code{"full"}, or \code{"minimal"}.}

  \item{rows}{Preferred terminal row count for graph rendering.}

  \item{cols}{Preferred terminal column count for graph rendering.}

  \item{path}{Optional workspace or \code{.gen} directory path for \code{Repository()}.}
}
\value{
\code{HashId()}, \code{Block()}, and \code{DbContext()}
return lightweight S3 objects. \code{GraphController()} and \code{Repository()}
return environment-backed helper objects with methods attached as fields.
}
\description{
These constructors provide the main R-side object model for \pkg{genr}.
\code{HashId()} and \code{Block()} are small value objects.
\code{DbContext()} resolves workspace and database paths.
\code{Repository()} exposes database and block-group helper methods, while
\code{GraphController()} exposes stateful rendering and navigation methods for a
single block group.
}
\details{
\code{Repository()} exposes the methods \code{execute(query)},
\code{query(query)}, \code{get_block_group_by_id(id)}, \code{get_block_groups()},
\code{get_block_groups_by_collection(collection_name)},
\code{block_group_to_dict(block_group)},
\code{block_group_to_rustworkx(block_group)},
\code{block_group_to_networkx(block_group)},
\code{plot(block_group, rows = NULL, cols = NULL, detail = "normal")},
\code{get_block_sequence(node_key)},
\code{stitch(bgs, new_sample, new_region)},
\code{build_index(bgs = NULL, sequence_kind = "dna", k = 16L)},
\code{search(query, bgs = NULL, sequence_kind = "dna")}, and
\code{clear_index(bgs = NULL)}.

\code{BlockGroup} objects expose the methods
\code{plot(rows = NULL, cols = NULL, detail = "normal")},
\code{export_fasta(filename)}, \code{export_gfa(filename, node_max = NULL)},
\code{export_genbank(filename)},
\code{build_index(sequence_kind = "dna", k = 16L)},
\code{search(query, sequence_kind = "dna")}, \code{clear_index()},
\code{subgraph(new_sample, start, end, backbone = NULL)}, and
\code{chunks(new_sample, breakpoints = NULL, chunk_size = NULL, backbone = NULL)}.

\code{GraphController()} exposes the methods \code{set_detail(detail)},
\code{render_frame(cols = ctrl$cols, rows = ctrl$rows)}, \code{zoom_in()},
\code{zoom_out()}, \code{handle_click(col, row)}, and \code{move_by(dx, dy)}.
}
\examples{
\dontrun{
init()

id <- HashId("abc123")
key <- Block(id, 1, 4)
ctx <- DbContext()

repo <- Repository()
groups <- repo$get_block_groups()

if (length(groups) > 0) {
  ctrl <- repo$plot(groups[[1]], rows = 12, cols = 40)
  ctrl$render_frame(40, 12)
}
}
}
