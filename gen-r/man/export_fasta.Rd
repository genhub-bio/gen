\name{export_fasta}
\alias{export_fasta}
\title{Export a Collection to FASTA}
\usage{
export_fasta(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = NULL
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{filename}{Output FASTA path.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Optional sample name to export.}
}
\value{
A character scalar containing the output FASTA path.
}
\description{
Exports a Gen collection or sample to FASTA.
}
\examples{
\dontrun{
tmp <- tempfile("genr-workspace-")
dir.create(tmp)
old <- setwd(tmp)
on.exit(setwd(old), add = TRUE)

init()
writeLines(c(">seq1", "ATCG"), "input.fa")
import_fasta(filename = "input.fa", name = "demo", sample = "sample-a")
export_fasta(filename = "output.fa", name = "demo", sample = "sample-a")
}
}
