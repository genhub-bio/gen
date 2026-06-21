# Agent Guidance

This file captures working conventions for agents in the Gen repository. It applies from the repository root unless a more specific `AGENTS.md` is added in a subdirectory.

## Startup Checklist

1. Read the local rules before editing Rust:
   ```bash
   cat .rules/*
   ```
2. Check worktree state with `git status --short`. Do not overwrite or revert user changes unless explicitly asked.
3. Skim the relevant module, tests, and fixtures before changing behavior.
4. If working on a branch, compare against the base branch when useful with `git diff main...HEAD`.

## Project Context

Gen is a Rust CLI and library for version control of genetic sequences. It stores genome-length sequences and variants as graph data, supports branching and merging of sequence repositories, and imports or exports common bioinformatics formats such as FASTA, GenBank, GFA, GAF, VCF, BED, GFF, and GTF.

The repository is a Cargo workspace. The root crate provides the `gen` CLI and core library facade; subcrates hold graph algorithms, persistence, annotations, schemas, diffing, layout, and TUI support. Python and R bindings live outside the default workspace members.

## Repository Map

- `src/`: root `gen` crate, CLI entrypoint, imports, updates, exports, views, patching, and operation management.
- `gen-core/`: core domain types and sequence graph model.
- `gen-models/`: SQLite persistence layer and migrations for operations and graph models.
- `gen-graph/`: graph utilities and traversal behavior.
- `gen-diff/`: graph and operation diffing.
- `gen-annotations/`: annotation parsing and related helpers.
- `gen-capnp-schemas/`: Cap'n Proto schema definitions and generated Rust bindings.
- `gen-tui/`, `gen-sugiyama/`: terminal UI and graph layout support.
- `gen-python/`: PyO3/maturin Python bindings and Jupyter widget assets.
- `gen-r/`: R package and Rust bridge crate.
- `fixtures/`: shared test data. Prefer adding small, focused fixtures.
- `tests/`: integration tests.
- `docs/`, `examples/`, `paper/`: user docs, worked examples, and manuscript assets.

## Model Nuances

These are extremely important details about data models.

- Nodes represent a sequence stored in the Database. A GraphNode represents all or part of a Node. `sequence_start` and `sequence_end` are python-indexed slices of the Sequence a node points at. For example, the sequence "AAATTT", the `GraphNode { ..., sequence_start: 3, sequence_end: 5}` would represent "TT". Thus, `sequence_start` and `sequence_end` are NOT coordinates in graph space. `sequence_end` - `sequence_start` can be used to derive the length of a node however.

## Rust Conventions

- Follow `.rules/rust-coding-style.mdc`. The workspace manifests currently use Rust edition 2024; prefer the manifest when it differs from older rule text.
- Keep functions to seven or fewer parameters. Group related values into structs when needed.
- Prefer references over ownership unless the callee consumes the value.
- Use newtypes for domain-specific identifiers or coordinates when that improves type safety.
- Prefer `From` implementations over direct `Into` implementations.
- Clone shared pointers with `Arc::clone(&value)` or `Rc::clone(&value)`.
- Avoid wildcard imports and local imports inside functions. Prefer explicit module-level imports.
- Prefer `core` over `alloc` over `std` where practical.
- Use explicit `pub use` re-exports in module roots when shaping public APIs.
- Use `#[expect(lint, reason = "...")]` rather than `#[allow(...)]`.
- `expect` messages for `Result` and `Option` should start with `should`.
- Keep comments sparse and useful. Place comments on their own line above the code they explain.
- Avoid shorthand identifiers (`bg`, `gn`, `vid`, `ci`, `src`/`tgt`, `nid`, `succ`, `deg`, `rem`, `aa`). Spell out the full word (`block_group`, `graph_node`, `virtual_id`, `chromosome_index`, `source`/`target`, `node_id`, `successor`, `degree`, `remaining`, `amino_acid`) even when it makes a line longer; match the fuller naming already used elsewhere in the same module rather than introducing a new abbreviation.
- No banner comments (lines of `---`, `===`, or similar dividers). No double blank lines.

## Persistence And Migrations

- `gen-models/migrations/core/` stores graph model migrations.
- `gen-models/migrations/operations/` stores operation tracking migrations.
- During active development, amend the migration where a table or column was introduced unless the task explicitly asks for an additive migration.
- Keep `up.sql` and `down.sql` paired and reversible when possible.
- Treat stored graph coordinates, path identifiers, sample names, and collection names as domain data. Preserve existing semantics and naming unless the task is explicitly a schema redesign.

## Generated Code And Assets

- Do not hand-edit generated Rust under `src/generated/` or `gen-capnp-schemas/src/generated/` unless the file itself indicates it is maintained manually.
- Cap'n Proto changes should start from the `.capnp` files in `gen-capnp-schemas/`; then regenerate through the normal build path.
- The Python and R widgets include committed generated JavaScript assets. If widget source changes, run the relevant Makefile target and commit the generated outputs that the target updates.

## Testing And Validation

Use the narrowest reliable validation for the change, then broaden when touching shared behavior.

- Format Rust with:
  ```bash
  cargo fmt
  ```
- Compile the default workspace:
  ```bash
  cargo check
  ```
- Run focused tests with:
  ```bash
  cargo test -q <test-name>
  ```
- Run the main workspace test suite with:
  ```bash
  cargo test --all-targets --all-features -q
  ```
- Run clippy with:
  ```bash
  cargo clippy --all-features --all-targets --no-deps
  ```
- For documentation-sensitive public API changes:
  ```bash
  cargo doc --no-deps --all-features
  ```
- For Python bindings:
  ```bash
  cargo test -q --manifest-path gen-python/Cargo.toml
  ```
- For R bindings, use the Makefile target when the environment has Docker and R tooling:
  ```bash
  make r-test
  ```

Some builds and tests need the Cap'n Proto compiler (`capnp`) installed. Python packaging targets may need `maturin`, and R targets may need Docker, R, and npm depending on the path exercised.

## CLI And Fixtures

- Prefer exercising behavior through CLI-level tests or small helper tests when changing imports, updates, exports, patching, or branch operations.
- Keep fixture files minimal and descriptive. Avoid replacing large biological fixtures unless the test genuinely needs real-world scale.
- When adding a new file format edge case, include the smallest input that captures the bug or feature.
- Commands documented in `docs/commands.md` should stay aligned with `clap` definitions and examples.

## Documentation

- Update `README.md`, `docs/`, or examples when user-facing commands, workflows, or supported formats change.
- Rust doc comments should begin with a concise summary. Add `# Arguments` for functions with three or more parameters and `# Errors` for fallible APIs.
- Public structs and enum variants should have clear field or variant documentation when exported.

## Collaboration Rules

- Keep changes scoped to the requested behavior.
- Preserve existing style and naming in the touched module.
- Avoid broad refactors unless they are needed to make the requested change correct.
- Report any validation command that could not be run, including the reason.
- Do not clean, delete, or regenerate unrelated files to make the worktree look tidy.
