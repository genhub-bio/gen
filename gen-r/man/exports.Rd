\name{export_fasta}
\alias{export_fasta}
\alias{export_gfa}
\alias{export_genbank}
\title{Export a Gen Collection}
\usage{
export_fasta(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = NULL
)

export_gfa(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample,
  node_max = NULL
)

export_genbank(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{filename}{Output filename.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Sample name to export.}

  \item{node_max}{Optional node count cap used by \code{export_gfa()}.}
}
\value{
Each function returns the output filename as a character scalar.
}
\description{
These functions write collection or sample state from a Gen database into
external formats including FASTA, GFA, and GenBank.
}
\examples{
\dontrun{
init()
import_fasta(filename = "reference.fa", name = "demo", sample = "reference")

export_fasta(filename = "out.fa", name = "demo", sample = "reference")
export_gfa(filename = "out.gfa", name = "demo", sample = "reference")
}
}
\seealso{
\code{\link{import_fasta}}, \code{\link{Repository}}
}
