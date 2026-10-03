\name{import_fasta}
\alias{import_fasta}
\alias{import_gfa}
\alias{import_genbank}
\alias{import_library_files}
\alias{import_library}
\title{Import Data into a Gen Collection}
\usage{
import_fasta(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = "sample",
  indexes = character()
)

import_gfa(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = "sample"
)

import_genbank(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample = "sample"
)

import_library_files(
  workspace_path = NULL,
  db_path = NULL,
  library_name,
  parts,
  library,
  name = NULL,
  sample = "sample"
)

import_library(
  workspace_path = NULL,
  db_path = NULL,
  library_name,
  parts_list,
  name = NULL,
  sample = NULL
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{filename}{Input filename for FASTA, GFA, or GenBank import.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Sample name to import into.}

  \item{indexes}{Optional paths or URIs for existing FASTA index files.}

  \item{library_name}{Library name for combinatorial library imports.}

  \item{parts}{Path to a FASTA file describing parts for \code{import_library_files()}.}

  \item{library}{Path to a CSV library layout file for \code{import_library_files()}.}

  \item{parts_list}{List of columns for in-memory library import. Each column is a named character vector or \code{XStringSet}.}
}
\value{
Each function returns a character scalar confirming the import path taken.
}
\description{
These functions ingest external sequence representations into a Gen
collection. Depending on the input format, they either create a new collection
or add additional samples and paths to an existing collection. Local FASTA
imports are stored in an indexed BGZF asset and read from that local archive. A
remote BGZF FASTA with usable ".fai" and ".gzi" indexes supplied through
"indexes" remains a remote reference, so sequence reads use its URL. Other
remote FASTA inputs are downloaded, compressed, indexed, and stored locally.
}
\examples{
\dontrun{
init()

import_fasta(filename = "reference.fa", name = "demo", sample = "reference")
import_gfa(filename = "graph.gfa", name = "demo", sample = "sample-a")
import_genbank(filename = "record.gb", name = "demo", sample = "sample-a")

parts_list <- list(
  c(`part-a` = "AAAA", `part-b` = "CCCC"),
  c(`part-c` = "GGGG", `part-d` = "TTTT")
)
import_library("library-a", parts_list, name = "demo", sample = "sample-a")
}
}
\seealso{
\code{\link{update_with_fasta}}, \code{\link{export_fasta}},
\code{\link{Repository}}
}
