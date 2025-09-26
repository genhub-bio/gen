# Technology Stack

## Core Technologies

- **Language**: Rust (2025 edition)
- **Database**: SQLite with rusqlite bindings
- **CLI Framework**: clap with derive features
- **Graph Processing**: petgraph for graph algorithms
- **Bioinformatics**: noodles crate for file format support (FASTA, VCF, GFF, etc.)
- **Python Bindings**: PyO3 with maturin for Python integration

## Build System

### Cargo Workspace Structure

- Main crate: `gen` (CLI and library)
- `gen-core`: Core data structures and utilities
- `gen-models`: Database models and operations
- `gen-graph`: Graph algorithms and operations

### Key Dependencies

- `rusqlite`: SQLite database operations with bundled SQLite
- `noodles`: Bioinformatics file format parsing
- `petgraph`: Graph data structures and algorithms
- `clap`: Command-line argument parsing
- `serde`: Serialization/deserialization
- `itertools`: Iterator utilities

## Common Commands

### Building

```bash
# Build all features
cargo build --all-features

# Release build
cargo build --release

# Cross-compile (example for Linux from macOS)
rustup target add x86_64-unknown-linux-gnu
cargo build --release --target=x86_64-unknown-linux-gnu
```

### Testing and Quality

```bash
# Run tests
cargo test

# Linting with all features
cargo clippy --all-features --all-targets --no-deps

# Format code
cargo fmt

# Generate documentation
cargo doc --no-deps --all-features

# Security audit
cargo deny check
```

### Python Bindings

```bash
# Create virtual environment and build Python bindings
make python

# Manual build
maturin develop --release --features python-bindings --features extension-module
```

### Docker

```bash
# Build Docker image
make docker-build
```

## Feature Flags

- `python-bindings`: Enable PyO3 Python bindings
- `extension-module`: Python extension module support
- `models`: Database models (default)
- `cli`: Command-line interface (default)
- `benchmark`: Benchmarking utilities

## Database Migrations

- Core migrations: `migrations/core/`
- Operations migrations: `migrations/operations/`
- Automatic migration on database connection

## Public APIs

### Rust Library API

Gen can be used as a Rust library by importing the main crate:

```rust
use gen::{get_connection, models, core, graph};
use gen_models::collection::Collection;
use gen_models::sample::Sample;
```

**Key modules available:**

- `gen::models`: Database models and operations (behind `models` feature flag)
- `gen::core`: Core utilities, configuration, error types
- `gen::graph`: Graph algorithms and data structures
- Database connections: `get_connection()` and `get_operation_connection()`

### Python Bindings

Enable Python bindings with the `python-bindings` feature:

```python
import gen

# Access Gen functionality through Python API
# Located in src/python_api/ modules:
# - block_group.rs: Block group operations
# - factory.rs: Object creation utilities
# - layouts.rs: Layout and visualization
# - repository.rs: Repository management
# - utils.rs: Utility functions
```

**Python API modules:**

- Block group operations and queries
- Repository management and initialization
- Layout generation for visualization
- Utility functions for data manipulation

### CLI as API

The Gen CLI can be used programmatically:

```bash
# Initialize repository
gen init

# Import sequences
gen import fasta --path sequences.fa --name my_collection

# Export in different formats
gen export gfa --collection my_collection --output graph.gfa

# Query sequences
gen get-sequence --name my_collection --graph chr1 --start 1000 --end 2000
```

### Database Direct Access

No direct SQL should be used outside of model code. All interactions with the
database should be mediated through gen-models.
