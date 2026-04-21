\name{init}
\alias{init}
\title{Initialise a Gen Workspace}
\usage{
init()
}
\value{
A character scalar confirming that the Gen workspace was initialized.
}
\description{
Initialises a \code{.gen/} workspace in the current working directory.
}
\examples{
\dontrun{
tmp <- tempfile("genr-workspace-")
dir.create(tmp)
old <- setwd(tmp)
on.exit(setwd(old), add = TRUE)

init()
}
}
