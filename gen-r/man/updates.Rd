\name{update_with_fasta}
\alias{update_with_fasta}
\alias{update_with_gfa}
\alias{update_with_gaf}
\alias{update_with_vcf}
\alias{update_with_genbank}
\alias{update_with_library_files}
\alias{update_with_library}
\alias{update_with_sequence}
\title{Update an Existing Gen Collection}
\usage{
update_with_fasta(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample,
  new_sample,
  region_name
)

update_with_gfa(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample,
  new_sample
)

update_with_gaf(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  csv,
  name = NULL,
  sample,
  parent_sample = NULL
)

update_with_vcf(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  genotype = NULL,
  sample = NULL,
  parent_samples = character(),
  in_place = FALSE
)

update_with_genbank(
  workspace_path = NULL,
  db_path = NULL,
  filename,
  name = NULL,
  sample,
  create_missing = FALSE
)

update_with_library_files(
  workspace_path = NULL,
  db_path = NULL,
  name = NULL,
  sample,
  new_sample,
  path_name,
  library,
  parts
)

update_with_library(
  workspace_path = NULL,
  db_path = NULL,
  name = NULL,
  sample = NULL,
  new_sample_name,
  path_name,
  parts_list
)

update_with_sequence(
  workspace_path = NULL,
  db_path = NULL,
  sequence,
  name = NULL,
  sample,
  new_sample,
  region_name,
  no_reference_path_update = FALSE
)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}

  \item{filename}{Filename for the update input.}

  \item{name}{Collection name. Required if no default collection is set.}

  \item{sample}{Existing sample that provides the source graph context.}

  \item{new_sample}{New sample name created by the update.}

  \item{region_name}{Target region or path name for sequence-based updates.}

  \item{csv}{Auxiliary CSV mapping file used by \code{update_with_gaf()}.}

  \item{parent_sample}{Optional parent sample name used by \code{update_with_gaf()}.}

  \item{genotype}{Optional genotype selector for \code{update_with_vcf()}.}

  \item{parent_samples}{Character vector of parent sample names for \code{update_with_vcf()}.}

  \item{in_place}{Whether \code{update_with_vcf()} should update in place rather than create a child sample.}

  \item{create_missing}{Whether \code{update_with_genbank()} should create missing records.}

  \item{path_name}{Path or region name updated by the library operations.}

  \item{library}{Library layout CSV for \code{update_with_library_files()}.}

  \item{parts}{Parts FASTA for \code{update_with_library_files()}.}

  \item{new_sample_name}{New sample name created by \code{update_with_library()}.}

  \item{parts_list}{List of columns for \code{update_with_library()}. Each column is a named character vector or \code{XStringSet}.}

  \item{sequence}{Literal replacement sequence for \code{update_with_sequence()}.}

  \item{no_reference_path_update}{Whether to skip the reference path update for \code{update_with_sequence()}.}
}
\value{
Each function returns a character scalar describing the completed update.
}
\description{
These functions derive new graph state from existing collections. Depending on
the input format, they can splice literal sequence, align graph files, apply
variant calls, or materialize combinatorial library edits into derived samples.
}
\examples{
\dontrun{
init()
import_fasta(filename = "reference.fa", name = "demo", sample = "reference")

update_with_sequence(
  sequence = "AAAA",
  name = "demo",
  sample = "reference",
  new_sample = "edited",
  region_name = "chr1:10-13"
)
}
}
\seealso{
\code{\link{import_fasta}}, \code{\link{derive_chunks}},
\code{\link{Repository}}
}
