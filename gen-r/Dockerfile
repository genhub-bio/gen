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

COPY . .

RUN Rscript -e "install.packages('testthat', repos = 'https://cloud.r-project.org')"

RUN R CMD INSTALL gen-r

RUN Rscript -e "library(testthat); testthat::test_dir('gen-r/tests/testthat')"

CMD ["Rscript", "-e", "library(genr); sessionInfo()"]
