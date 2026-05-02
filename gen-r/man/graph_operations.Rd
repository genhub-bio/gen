\name{derive_chunks}
\alias{derive_chunks}
\alias{derive_subgraph}
\alias{make_stitch}
\title{Graph-Derived Collection Operations}
\usage{
derive_chunks(
  workspace_path = NULL,
  db_path = NULL,
  name = NULL,
  sample,
  new_sample,
  region,
  backbone = NULL,
  breakpoints = NULL,
  chunk_size = NULL
)

derive_subgraph(
  workspace_path = NULL,
  db_path = NULL,
  name = NULL,
  sample,
  new_sample,
  region,
  backbone = NULL
)

make_stitch(
  workspace_path = NULL,
  db_path = NULL,
  name = NULL,
  sample,
  new_sample,
  regions,
  new_region
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Existing sample used as the source graph.}

  \item{new_sample}{Derived sample name to create.}

  \item{region}{Region string to chunk or extract.}

  \item{backbone}{Optional backbone path or sample to anchor the operation.}

  \item{breakpoints}{Comma-separated breakpoints for \code{derive_chunks()}.}

  \item{chunk_size}{Alternative chunk size for \code{derive_chunks()}.}

  \item{regions}{Comma-separated region names passed to \code{make_stitch()}.}

  \item{new_region}{Name of the stitched output region.}
}
\value{
Each function returns a character scalar describing the derived graph operation.
}
\description{
These helpers derive new graph structure from existing regions. They can split a
region into chunks, extract a focused subgraph, or stitch multiple derived
regions together into a new path.
}
\examples{
\dontrun{
derive_chunks(
  name = "demo",
  sample = "reference",
  new_sample = "chunked",
  region = "chr1",
  breakpoints = "100,200"
)
}
}
\seealso{
\code{\link{update_with_sequence}}, \code{\link{Repository}}
}
