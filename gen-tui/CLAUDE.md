# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gen is a version control system for genetic sequences that uses a graph-based data model to efficiently store genome-length sequences and variations. The system supports polyploid genomes, pooled genotypes, and complex biological variations through a three-layer architecture.

## Development Commands

### Building and Testing
- `source .venv/bin/activate` - Activate Python virtual environment (required for first build in new session)
- `cargo build --all-features` - Build the project with all features
- `git add .` - Stage changes before formatting (rustfmt may modify files)
- `cargo fmt --all --` - Format code with rustfmt (run before commits to avoid hook conflicts)
- `git commit` - Commit to git with a meaningful but concise commit message (no emoji)
- `cargo test` - Run test suite
- `cargo build --release` - Build optimized release version



### Running Tests
- Tests are distributed throughout the codebase in `tests.rs` files
- Widget system tests use snapshot testing with `insta` crate
- Use `cargo test` for standard unit tests

## Architecture Overview

### Core Data Model
The system uses a **block graph model** (additive) rather than a segment graph model, where:
- **Nodes** represent sequence fragments with stable IDs
- **Edges** define connections between fragments
- **Paths** define linear sequences by walking through edges
- **Collections** organize molecules (chromosomes, proteins, etc.)
- **Block groups** represent individual chromosomes/contigs with name/sample/collection facets

### Three-Layer Widget System (`src/views/widget/`)

The visualization system implements a sophisticated three-layer architecture for rendering large graphs:

#### 1. Domain Layer
- Original database graph data (e.g., `DiGraphMap<GraphNode, GraphEdge>`)
- Contains domain-specific node/edge data with biological meaning
- Examples: `GraphNode { node_id, sequence_start, sequence_end, ... }`

#### 2. Partition Layer (`partition_table.rs`, `partition_controller.rs`)
- Splits large graphs at articulation points for memory management
- Uses `StableGraph<PartitionNode, PartitionEdge>` with stitch nodes for boundaries
- Handles coordinate stitching via Fenwick trees for efficient spatial queries
- Manages dynamic loading/unloading based on viewport

#### 3. Layout Layer (`layout.rs`)
- Pure spatial positioning using Sugiyama hierarchical layout algorithm
- `StableGraph<LayoutNode, LayoutEdge>` with world coordinates
- R-tree spatial indexing for viewport culling
- Support for multiple levels of detail (Minimal/Full/Truncated)

### Domain Agnosticism
The widget system works with any directed acyclic graph via two customization traits:
- `NodeSizer<G>` - Determines visual dimensions for all nodes in a partition
- `NodeRenderer<G>` - Handles domain-specific visual rendering of the nodes in the viewport
Both are also dependent on the requested level of visual detail (minimal, full, truncated).

### Key Components
- `GraphController` - Central coordinator, viewport management, animations
- `PartitionController` - Dynamic partition loading with LRU eviction  
- `LayoutEngine` - Sugiyama algorithm implementation
- `GraphWidget` - Ratatui integration with theme support

## Database and Models

### Core Models (`src/models/`)
- `Node`, `Edge` - Graph structure components
- `Path`, `PathEdge` - Linear sequence representations  
- `BlockGroup`, `Collection`, `Sample` - Organization hierarchy
- `Operations` - Version control and change tracking
- `Sequence`, `Strand` - Biological sequence data

### Database
- SQLite with bundled driver
- Migrations handled by `rusqlite_migration` from `migrations/` directory
- Connection management in model implementations

## File Structure

TODO: redo this section

## Testing Architecture

### Widget Testing (`src/views/widget/testing/`)
- **LayoutTester** - Test individual layouts with snapshot testing
- **TestUtils** - Common testing patterns and verification methods
- **Mocks** - Standardized test graphs, sizers, and renderers
- Uses `insta` crate for visual regression testing
- TestBackend integration with ratatui for terminal output testing

### Test Structure
- Unit tests alongside implementation files
- Integration tests use standardized mock infrastructure
- Snapshot tests for visual components using `insta::assert_snapshot!`

## Important Coding Patterns

### Error Handling
- Uses `thiserror` for custom error types
- Consistent error propagation with `?` operator

### Graph Operations
- Leverage `petgraph` crate with `DiGraphMap` and `StableGraph` types
- Domain-agnostic algorithms through trait bounds
- Efficient spatial queries using R-tree indexing (`rstar` crate)

### Widget Development
- Implement `NodeSizer<G>` and `NodeRenderer<G>` traits for new domains
- Use `WorldBuffer` for coordinate-based rendering
- Support multiple levels of detail (Minimal/Full/Truncated) in size and render methods

### Performance Considerations
- Partition-based loading for large graphs
- Spatial indexing with viewport culling
- Lazy computation of layouts
- LRU cache eviction for memory management


Commit to git only when the users asks to commit to git. Do this by first staging changes with `git add .`, running `cargo fmt --all --`, staging any edits made by cargo fmt, running `cargo clippy --all-targets --all-features`, resolving ALL warnings (pre-commit hook will fail otherwise), and committing with a meaningful but concise commit message (no emoji)

**IMPORTANT**: All clippy warnings must be resolved before committing or the pre-commit hook will fail the commit. If you don't know how to resolve a warning:
- Suggest adding `#[allow(clippy::warning_name)]` or `#[allow(unused_variables)]` annotations
- Suggest commenting out problematic sections temporarily
- Ask the user what to do with the specific warning

