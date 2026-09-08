# syntax=docker/dockerfile:1
FROM scratch AS sccache

FROM rust:1.95-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    capnproto \
    clang \
    libclang-dev \
    mold \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libuv1-dev \
    pkg-config \
    pandoc \
    r-base \
    r-base-dev \
    r-bioc-biostrings \
    r-bioc-s4vectors \
    r-cran-testthat \
    r-cran-knitr \
    r-cran-rmarkdown \
    r-cran-remotes \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/gen

COPY Cargo.toml Cargo.lock LICENSE ./
COPY .cargo .cargo
COPY theme ./theme
COPY src ./src
COPY gen-core ./gen-core
COPY gen-diff ./gen-diff
COPY gen-graph ./gen-graph
COPY gen-models ./gen-models
COPY gen-models-macros ./gen-models-macros
COPY gen-tui ./gen-tui
COPY gen-sugiyama ./gen-sugiyama
COPY gen-annotations ./gen-annotations
COPY gen-capnp-schemas ./gen-capnp-schemas
COPY gen-r ./gen-r

RUN --mount=type=secret,id=github_pat,required=false \
    GITHUB_PAT=$(cat /run/secrets/github_pat 2>/dev/null || true) Rscript -e "remotes::install_github('keller-mark/anyhtmlwidget@be23d31587b3b0986febef00ba27567e46dc3338', upgrade='never')"

RUN --mount=type=bind,from=sccache,target=/opt/sccache,ro \
    --mount=type=secret,id=actions_results_url,required=false \
    --mount=type=secret,id=actions_runtime_token,required=false \
    if [ -x /opt/sccache/sccache ] \
        && [ -s /run/secrets/actions_results_url ] \
        && [ -s /run/secrets/actions_runtime_token ]; then \
        ACTIONS_CACHE_SERVICE_V2=on \
        ACTIONS_RESULTS_URL="$(cat /run/secrets/actions_results_url)" \
        ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/actions_runtime_token)" \
        RUSTC_WRAPPER=/opt/sccache/sccache \
        SCCACHE_GHA_ENABLED=true \
        R CMD INSTALL gen-r; \
    else \
        R CMD INSTALL gen-r; \
    fi
RUN Rscript -e "tools::buildVignettes(dir = 'gen-r', tangle = FALSE)"

CMD ["Rscript", "-e", "library(testthat); testthat::test_dir('gen-r/tests/testthat')"]
