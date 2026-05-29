\name{genr-repository-helpers}
\title{Repository Access and Graph Rendering Helpers}
\description{
\code{Repository()} opens a Gen database and exposes higher-level helper methods
for querying block groups, executing SQL, extracting graph dictionaries, and
rendering block groups through \code{GenPlot()}.
}
\details{
\code{Repository()} returns an environment with at least these fields:
\code{gen_dir}, \code{db_path}, \code{execute}, \code{query},
\code{get_block_group_by_id}, \code{get_block_groups},
\code{get_block_groups_by_collection}, \code{block_group_to_dict},
\code{plot}, \code{get_block_sequence}, \code{stitch},
\code{build_index}, \code{search}, and \code{clear_index}.

\code{BlockGroup} objects expose \code{plot}, \code{export_fasta},
\code{export_gfa}, \code{export_genbank}, \code{build_index}, \code{search},
\code{clear_index}, \code{subgraph}, and \code{chunks}.

\code{GenPlot()} returns a \code{gen_plot} environment with the fields \code{db_path},
\code{block_group_id}, \code{detail}, \code{ops}, \code{rows}, and
\code{cols}, plus the methods \code{set_detail}, \code{render_frame},
\code{zoom_in}, \code{zoom_out}, \code{handle_click}, and \code{move_by}.
}
\seealso{
\code{\link{HashId}}, \code{\link{derive_chunks}}
}
