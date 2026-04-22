FROM rust:1.90-bookworm

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

RUN Rscript -e "install.packages('testthat', repos = 'https://cloud.r-project.org')"

RUN R CMD INSTALL gen-r

CMD ["Rscript", "-e", "library(testthat); testthat::test_dir('gen-r/tests/testthat')"]
