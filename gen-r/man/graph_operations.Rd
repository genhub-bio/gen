\name{derive_chunks}
\alias{derive_chunks}
\alias{derive_subgraph}
\alias{stitch}
\title{Graph-Derived Collection Operations}
\usage{
derive_chunks(
  collection_name = NULL,
  sample,
  new_sample,
  region,
  backbone = NULL,
  breakpoints = NULL,
  chunk_size = NULL
)

derive_subgraph(
  collection_name = NULL,
  sample,
  new_sample,
  region,
  backbone = NULL
)

stitch(
  collection_name = NULL,
  sample,
  new_sample,
  regions,
  new_region
)
}
\arguments{
  \item{collection_name}{Collection name. Required if no default collection is set.}

  \item{sample}{Existing sample used as the source graph.}

  \item{new_sample}{Derived sample name to create.}

  \item{region}{Region string to chunk or extract (e.g. \code{"chr1"} or \code{"chr1:100-200"}).}

  \item{backbone}{Optional backbone path name to anchor the operation.}

  \item{breakpoints}{Comma-separated breakpoints for \code{derive_chunks()}.}

  \item{chunk_size}{Chunk size in bases as an alternative to \code{breakpoints}.}

  \item{regions}{Comma-separated region names passed to \code{stitch()}.}

  \item{new_region}{Name of the stitched output region.}
}
\value{
Each function returns a character scalar describing the derived graph operation.
}
\description{
These helpers derive new graph structure from existing regions. They can split a
region into chunks, extract a focused subgraph, or stitch multiple derived
regions together into a new path.

The preferred interface is through a \code{\link{Repository}} object:
\code{repo$derive_chunks()}, \code{repo$derive_subgraph()}, and
\code{repo$stitch()}.  The standalone functions here are lower-level wrappers
that operate on an open repository identified by \code{collection_name}.
}
\examples{
\dontrun{
repo <- Repository()
repo$import_fasta("ref.fa", sample = "reference")

# Split into coordinate chunks
repo$derive_chunks(
  sample = "reference", new_sample = "chunked",
  region = "chr1", breakpoints = "100,200"
)

# Stitch the chunks back together
chunks <- Filter(
  function(bg) bg$sample_name == "chunked",
  repo$get_block_groups()
)
repo$stitch(chunks, new_sample = "stitched", new_region = "chr1.stitched")
}
}
\seealso{
\code{\link{update_with_sequence}}, \code{\link{Repository}}
}
