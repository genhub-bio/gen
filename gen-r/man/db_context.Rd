\name{db_context}
\alias{db_context}
\title{Resolve a Gen Database Context}
\usage{
db_context(workspace_path = NULL, db_path = NULL)
}
\arguments{
  \item{workspace_path}{Optional path to the workspace root.}

  \item{db_path}{Optional path to a specific SQLite database.}
}
\value{
A named list with \code{workspace_path} and \code{db_path}.
}
\description{
Validates that Gen can open the workspace and target database, then returns the
resolved workspace and database paths.
}
\examples{
\dontrun{
tmp <- tempfile("genr-workspace-")
dir.create(tmp)
old <- setwd(tmp)
on.exit(setwd(old), add = TRUE)

init()
db_context()
}
}
