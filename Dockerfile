FROM rust:1.95-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    capnproto \
    clang \
    git \
    libclang-dev \
    mold \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gen

COPY Cargo.lock Cargo.toml LICENSE ./
COPY .cargo .cargo
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

RUN cargo build --release
