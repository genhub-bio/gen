# Gen - Genetic Sequence Version Control System

Gen is a version control system for genetic sequences that efficiently stores genome-length sequences and sequence variations. It provides native support for polyploid genomes and pooled genotypes using a graph-based data model.

## Core Concepts

- **Block Graph Model**: Sequences are represented as networks of nodes (sequence fragments) connected by edges, allowing additive modifications without splitting existing nodes
- **Collections**: Groups of sequences representing chromosomes, proteomes, or DNA mixtures
- **Samples**: Real individuals or virtual experimental outcomes
- **Paths**: Linear sequence reconstruction by walking through nodes and edges
- **Branches**: Parallel development tracks for exploring modifications without affecting main project

## Key Features

- Import/export standard sequence formats (FASTA, GenBank, GFA, VCF, GAF)
- Version control with branching and merging capabilities
- Support for complex biological variations and engineering
- Command-line interface with subcommands for all operations
- Python bindings for programmatic access
- Cross-platform support with prebuilt binaries

## Target Use Cases

- Genome assembly and variation analysis
- Combinatorial genetic design and cloning
- Strain crosses and breeding programs
- Iterative genetic engineering workflows