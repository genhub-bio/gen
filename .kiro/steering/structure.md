# Project Structure

## Root Directory Layout

```
gen/
├── src/                    # Main crate source code
├── gen-core/              # Core utilities and data structures
├── gen-models/            # Database models and operations
├── gen-graph/             # Graph algorithms and operations
├── python/                # Python bindings and API
├── migrations/            # Database migration scripts
├── examples/              # Usage examples and workflows
├── docs/                  # Documentation and figures
├── fixtures/              # Test data and sample files
├── config/                # Configuration files (themes)
└── paper/                 # Research paper and assets
```

## Source Code Organization

### Main Crate (`src/`)
- `main.rs`: CLI entry point
- `lib.rs`: Library exports and common utilities
- `commands/`: CLI command implementations
  - `import/`: Data import commands (FASTA, GenBank, GFA, etc.)
  - `export/`: Data export commands
  - `update/`: Data update operations
- `imports/`: File format import logic
- `exports/`: File format export logic
- `updates/`: Update operations (VCF, GAF, etc.)
- `views/`: Data visualization and display
- `annotations/`: Annotation handling (GFF)
- `translate/`: Coordinate translation utilities
- `python_api/`: Python binding implementations

### Workspace Crates
- **gen-core**: Core types, configuration, error handling
- **gen-models**: Database models, migrations, operations
- **gen-graph**: Graph data structures and algorithms

## Key Modules

### Commands Structure
Each command follows a consistent pattern:
- Import commands: `src/commands/import/{format}.rs`
- Export commands: `src/commands/export/{format}.rs`
- Update commands: `src/commands/update/{format}.rs`

### File Format Support
- FASTA: Sequence import/export
- GenBank: Annotated sequence handling
- GFA: Graph format assembly
- VCF: Variant call format
- GAF: Graph alignment format
- BED/GFF: Annotation formats

### Database Layer
- Models in `gen-models/src/`
- Migrations in `migrations/{core,operations}/`
- Automatic schema management

## Configuration and Data

### User Data Locations
- `.gen/`: Repository metadata and configuration
- Database files: SQLite databases for sequence storage
- Default database: `~/.gen/default.db` or specified path

### Test and Example Data
- `fixtures/`: Test data for various file formats
- `examples/`: Real-world usage examples
  - Combinatorial plasmid design
  - Human variation analysis
  - Yeast strain crosses

## Development Patterns

### Module Organization
- Each major feature has its own module
- Consistent import/export/update pattern
- Separation of CLI logic from core functionality
- Database operations abstracted through models

### Error Handling
- Custom error types in each crate
- Consistent error propagation using `thiserror`
- Database transaction management

### Testing
- Unit tests alongside source code
- Integration tests in `tests/` directories
- Test helpers in `test_helpers.rs` modules
- Fixture data for comprehensive testing