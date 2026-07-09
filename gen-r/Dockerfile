# syntax=docker/dockerfile:1
FROM rust:1.95-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    capnproto \
    libclang-dev \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libuv1-dev \
    pkg-config \
    pandoc \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/gen

COPY Cargo.toml Cargo.lock LICENSE ./
COPY theme ./theme
COPY src ./src
COPY gen-core ./gen-core
COPY gen-diff ./gen-diff
COPY gen-graph ./gen-graph
COPY gen-models ./gen-models
COPY gen-tui ./gen-tui
COPY gen-sugiyama ./gen-sugiyama
COPY gen-annotations ./gen-annotations
COPY gen-capnp-schemas ./gen-capnp-schemas
COPY gen-r ./gen-r

RUN Rscript -e "install.packages(c('testthat', 'knitr', 'rmarkdown'), repos = 'https://cloud.r-project.org')"
RUN --mount=type=secret,id=github_pat,required=false \
    GITHUB_PAT=$(cat /run/secrets/github_pat 2>/dev/null || true) Rscript -e "if (!requireNamespace('remotes', quietly=TRUE)) install.packages('remotes', repos='https://cloud.r-project.org'); remotes::install_github('keller-mark/anyhtmlwidget@be23d31587b3b0986febef00ba27567e46dc3338', upgrade='never')"

RUN R CMD INSTALL gen-r
RUN Rscript -e "tools::buildVignettes(dir = 'gen-r', tangle = FALSE)"

CMD ["Rscript", "-e", "library(testthat); testthat::test_dir('gen-r/tests/testthat')"]
