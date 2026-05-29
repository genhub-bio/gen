\name{HashId}
\alias{HashId}
\alias{GenPlot}
\alias{Repository}
\title{Constructors for Gen R Objects}
\usage{
HashId(hash_id)

GenPlot(
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

  \item{db_path}{Path to a Gen SQLite database.}

  \item{block_group_id}{Block group hash ID string or \code{gen_hash_id} object.}

  \item{detail}{Graph detail level: \code{"normal"}, \code{"full"}, or \code{"minimal"}.}

  \item{rows}{Preferred terminal row count for graph rendering.}

  \item{cols}{Preferred terminal column count for graph rendering.}

  \item{path}{Optional workspace or \code{.gen} directory path for \code{Repository()}.}
}
\value{
\code{HashId()} returns a lightweight \code{gen_hash_id} S3 object.
\code{GenPlot()} and \code{Repository()} return environment-backed
helper objects with methods attached as fields.
}
\description{
These constructors provide the main R-side object model for \pkg{genr}.
\code{HashId()} is a small value object wrapping a hash string.
\code{Repository()} opens a Gen workspace and exposes database and
sequence-graph helper methods. \code{GenPlot()} exposes stateful
rendering and navigation methods for a single sequence graph.

Graph nodes (\code{gen_node}) are obtained directly from
\code{graph_dict$nodes[[i]]$key} and passed to
\code{repo$get_node_sequence()}.
}
\details{
\code{Repository()} exposes the methods \code{execute(query)},
\code{query(query)}, \code{get_block_group_by_id(id)}, \code{get_block_groups()},
\code{get_block_groups_by_collection(collection_name)},
\code{block_group_to_dict(block_group)},
\code{plot(block_group, rows = NULL, cols = NULL, detail = "normal")},
\code{get_node_sequence(node)},
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

\code{GenPlot()} exposes the methods \code{set_detail(detail)},
\code{render_frame(cols = ctrl$cols, rows = ctrl$rows)}, \code{zoom_in()},
\code{zoom_out()}, \code{handle_click(col, row)}, and \code{move_by(dx, dy)}.
}
\examples{
\dontrun{
id <- HashId("abc123")

repo <- Repository()
groups <- repo$get_block_groups()

if (length(groups) > 0) {
  ctrl <- repo$plot(groups[[1]], rows = 12, cols = 40)
  ctrl$render_frame(40, 12)

  graph_dict <- groups[[1]]$.inner$to_dict()
  node_seq <- repo$get_node_sequence(graph_dict$nodes[[1]]$key)
}
}
}
