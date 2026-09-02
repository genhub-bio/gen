# gen-models-macros

`gen-models-macros` provides the `ModelSelect` derive used by `gen-models` to create typed,
fluent SQLite selectors for persisted models. It replaces handwritten query-parameter structs and
string-based ordering with methods and field constants generated from the model itself.

This crate is intended to be used through `gen-models`, which re-exports `ModelSelect` and provides
the runtime selector implementation.

## Basic usage

Derive `ModelSelect` on a named-field struct that also implements
[`Query`](../gen-models/src/traits.rs):

```rust
use gen_models::{Direction, ModelSelect, traits::Query};
use rusqlite::Row;

#[derive(Debug, PartialEq, ModelSelect)]
pub struct Sample {
    pub name: String,
    pub is_reference: bool,
}

impl Query for Sample {
    type Model = Sample;

    const PRIMARY_KEY: &'static str = "name";
    const TABLE_NAME: &'static str = "samples";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            name: row.get(0).expect("should read sample name"),
            is_reference: row.get(1).expect("should read reference flag"),
        }
    }
}
```

The derive adds `Sample::select(conn)` and generates `SampleSelect`:

```rust
let samples = Sample::select(conn)
    .name_contains("foo")
    .is_reference(true)
    .order_by(SampleSelect::Name, Direction::Asc)
    .limit(10)
    .offset(3)
    .load();
```

`load()` returns `Vec<Sample>`. Filter values, history refs, limits, and offsets are passed to
SQLite as bound parameters rather than interpolated into SQL.

## Generated API

For a model named `Sample`, the derive generates:

- `SampleSelect<'conn>`, the fluent query builder.
- `SampleSelectField`, an enum containing one variant for every selectable field.
- Associated field constants such as `SampleSelect::Name`, which provide the concise ordering API.
- `Sample::select(conn)`, the preferred selector constructor.

Every selectable field receives an exact-match method:

```rust
Sample::select(conn).is_reference(true);
```

`String` fields also receive a case-insensitive `_contains` method:

```rust
Sample::select(conn).name_contains("reference");
```

`Option<T>` fields receive an `_is_null` method in addition to an exact-match method that accepts
`T`:

```rust
let children = BlockGroup::select(conn)
    .parent_block_group_id(parent_id)
    .load();

let roots = BlockGroup::select(conn)
    .parent_block_group_id_is_null()
    .load();
```

## Ordering and pagination

Ordering uses generated field constants, so misspelled or removed column names fail at compile
time:

```rust
let samples = Sample::select(conn)
    .order_by(SampleSelect::Name, Direction::CaseInsensitiveAsc)
    .order_by(SampleSelect::IsReference, Direction::Desc)
    .limit(25)
    .offset(50)
    .load();
```

Available directions are:

- `Direction::Asc`
- `Direction::Desc`
- `Direction::CaseInsensitiveAsc`
- `Direction::CaseInsensitiveDesc`

An offset without an explicit limit is supported; the runtime renderer emits SQLite's unlimited
`LIMIT -1` form before the offset.

## Historical queries

Use `with_ref` to read a Dolt history ref. Omitting `with_ref` reads the current working state.

```rust
let historical_samples = Sample::select(conn)
    .with_ref("main~1")
    .name_contains("foo")
    .load();
```

For the default source, a historical query reads `dolt_at_<table>(:history_ref)`. When selectors
are joined, the same history ref is applied to every source. If both selectors specify refs, they
must be equal.

Models that set `Query::HISTORY_TABLE_NAME` to `None` cannot use `with_ref`.

## Joins

Selectors can join another model selector and carry its filters into the resulting query:

```rust
let samples = Sample::select(conn)
    .name_contains("sample")
    .join(BlockGroup::select(conn).collection_name("example"))
    .order_by(SampleSelect::Name, Direction::Asc)
    .load();
```

The returned rows are always instances of the selector on the left, so this query returns
`Vec<Sample>`.

Join conditions are inferred at runtime from `PRAGMA foreign_key_list`. The tables must have
exactly one unambiguous direct foreign-key relationship. Both relationship directions and
composite foreign keys are supported. Column identifiers obtained from the database schema are
quoted before they are added to SQL.

A related source alias can only be joined once. Put every filter for that related model on one
selector:

```rust
let paths = Path::select(conn)
    .join(
        BlockGroup::select(conn)
            .collection_name("example")
            .sample_name("reference"),
    )
    .load();
```

Do not express the same query as two joins:

```rust,ignore
// Panics because both methods attempt to join the `block_groups` alias.
Path::select(conn)
    .collection_name("example")
    .sample_name("reference")
    .load();
```

Joined selectors must:

- Use the same database connection.
- Use the same historical ref, when explicitly set on both selectors.
- Have unique source aliases.
- Leave `limit` and `offset` unset; apply pagination to the outer selector after the join.

Violating these requirements, or attempting an ambiguous or unrelated join, currently panics with
a descriptive message.

## Configuration attributes

Fields use their Rust field name as the SQL column name by default. Override or omit a generated
field with `model_select` attributes:

```rust
#[derive(ModelSelect)]
pub struct Example {
    #[model_select(column = "display_name")]
    pub name: String,

    #[model_select(skip)]
    pub derived_value: String,
}
```

The derive also accepts advanced struct-level options:

```rust,ignore
#[derive(ModelSelect)]
#[model_select(
    alias = "sample_rows",
    source = sample_source,
    select = "sample_rows.name, sample_rows.is_reference"
)]
pub struct Sample {
    pub name: String,
    pub is_reference: bool,
}
```

- `alias = "..."` sets the SQL alias used to qualify generated columns.
- `source = path::to::function` supplies a `fn(Option<&str>) -> String` that renders the `FROM`
  source. Its SQL must expose the configured alias and honor the optional history ref when needed.
- `select = "..."` replaces the default `<alias>.*` select list. Its column order must still match
  `Query::process_row`.

These attributes contain trusted, compile-time SQL metadata. They must never be populated from
runtime input.

## How it works

The implementation is split between compile-time generation and runtime execution.

At compile time, `ModelSelect`:

1. Accepts a non-generic struct with named fields.
2. Reads the model and field-level `model_select` attributes.
3. Generates the selector struct, typed ordering fields, filter methods, and `SelectQuery`
   implementation.
4. Generates `Model::select(conn)` as the public entry point.

At runtime, [`gen-models::select`](../gen-models/src/select.rs):

1. Stores generated filters, ordering, joins, pagination, and the optional history ref.
2. Infers requested joins from the open database's foreign-key metadata.
3. Renders one `SELECT` statement for the base model and its joined sources.
4. Binds every runtime value through `rusqlite`.
5. Maps result rows through the base model's `Query::process_row` implementation.

The runtime support cannot live in this proc-macro crate. A proc-macro executes inside the compiler
and can export procedural macros, but it cannot export the normal reusable runtime types needed by
all generated selectors. It also has no access to the query's live database connection. Keeping
the SQL renderer in ordinary Rust provides one implementation that can be tested and reviewed
without expanding every derive.

## Current limitations

- Only structs with named fields are supported.
- Generic model structs are not supported.
- At least one field must remain selectable after applying `skip`.
- Joins require one direct, unambiguous foreign-key relationship in the live schema.
- The same source alias cannot be joined more than once.
- `load()` returns `Vec<Model>` and currently panics on SQL preparation, execution, or row-mapping
  failures rather than returning a `Result`.
- Generated code references `gen_models`, so downstream consumers should use the macro through the
  `gen-models` crate under its standard crate name.
