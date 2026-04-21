\name{get_gen_dir}
\alias{get_gen_dir}
\title{Return the Active Gen Directory}
\usage{
get_gen_dir()
}
\value{
A character scalar containing the path to the active \code{.gen} directory.
}
\description{
Returns the path to the current workspace's \code{.gen} directory, creating it
if needed.
}
\examples{
\dontrun{
tmp <- tempfile("genr-workspace-")
dir.create(tmp)
old <- setwd(tmp)
on.exit(setwd(old), add = TRUE)

init()
get_gen_dir()
}
}
