\name{genr-repository-helpers}
\title{Repository Access and Graph Rendering Helpers}
\description{
\code{Repository()} opens a Gen database and exposes higher-level helper methods
for querying sequence graphs, executing SQL, extracting graph dictionaries, and
rendering sequence graphs through \code{GenPlot()}.
}
\details{
\code{Repository()} returns an environment with at least these fields:
\code{gen_dir}, \code{db_path}, \code{execute}, \code{query},
\code{get_sequence_graph_by_id}, \code{get_sequence_graphs},
\code{get_sequence_graphs_by_collection},
\code{get_node_sequence}, \code{stitch},
\code{build_index}, \code{search}, and \code{clear_index}.

\code{SequenceGraph} objects expose \code{plot}, \code{export_fasta},
\code{export_gfa}, \code{export_genbank}, \code{build_index}, \code{search},
\code{clear_index}, \code{subgraph}, and \code{chunks}.

\code{GenPlot()} returns a \code{gen_plot} environment with the fields \code{db_path},
\code{sequence_graph_id}, \code{detail}, \code{ops}, \code{rows}, and
\code{cols}, plus the methods \code{set_detail}, \code{render_frame},
\code{zoom_in}, \code{zoom_out}, \code{handle_click}, and \code{move_by}.
}
\seealso{
\code{\link{HashId}}, \code{\link{derive_chunks}}
}
