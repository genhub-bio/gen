\name{import_fasta}
\alias{import_fasta}
\title{Import a FASTA File}
\usage{
import_fasta(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = "sample",
  shallow = FALSE
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{filename}{Path to the FASTA file to import.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Sample name to import into.}

  \item{shallow}{Whether to store sequence data by reference instead of inline.}
}
\value{
A character scalar confirming that the FASTA import completed.
}
\description{
Imports a FASTA file into a Gen collection.
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
}
}
