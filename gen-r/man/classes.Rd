\name{HashId}
\alias{HashId}
\alias{GenPlot}
\alias{Repository}
\title{Constructors for Gen R Objects}
\usage{
HashId(hash_id)

GenPlot(
  db_path,
  sequence_graph_id,
  detail = "normal",
  rows = NULL,
  cols = NULL
)

Repository(path = NULL)
}
\arguments{
  \item{hash_id}{Hash identifier string.}

  \item{db_path}{Path to a Gen SQLite database.}

  \item{sequence_graph_id}{Sequence graph hash ID string or \code{gen_hash_id} object.}

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
\code{to_dict()$nodes[[i]]$key} and passed to
\code{repo$get_node_sequence()}.
}
\details{
\code{Repository()} exposes the methods \code{execute(query)},
\code{query(query)}, \code{get_sequence_graph_by_id(id)}, \code{get_sequence_graphs()},
\code{get_sequence_graphs_by_collection(collection_name)},
\code{get_node_sequence(node)},
\code{stitch(sgs, new_sample, new_region)},
\code{build_index(sgs = NULL, sequence_kind = "dna", k = 16L)},
\code{search(query, sgs = NULL, sequence_kind = "dna")}, and
\code{clear_index(sgs = NULL)}.

\code{SequenceGraph} objects expose the methods
\code{plot(rows = NULL, cols = NULL, detail = "normal")},
\code{export_fasta(filename)}, \code{export_gfa(filename, node_max = NULL)},
\code{export_genbank(filename)},
\code{build_index(sequence_kind = "dna", k = 16L)},
\code{search(query, sequence_kind = "dna")}, \code{get_node_sequence(node)},
\code{clear_index()},
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
graphs <- repo$get_sequence_graphs()

if (length(graphs) > 0) {
  p <- plot(graphs[[1]])

  graph_dict <- graphs[[1]]$to_dict()
  node_seq <- repo$get_node_sequence(graph_dict$nodes[[1]]$key)
}
}
}
