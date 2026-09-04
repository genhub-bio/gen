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
    #[model_select(primary_key)]
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
    .load()
    .expect("should load samples");
```

`load()` returns `Result<Vec<Sample>, ModelSelectError>`. Filter values, history refs, limits, and
offsets are passed to SQLite as bound parameters rather than interpolated into SQL.

## Generated API

For a model named `Sample`, the derive generates:

- `SampleSelect<'conn>`, the fluent query builder.
- Typed field constants such as `SampleSelect::Name`. Each constant carries its model, SQL column,
  and Rust value type.
- `Sample::select(conn)`, the preferred selector constructor.
- `Sample::all(conn)`, a convenience wrapper around `Sample::select(conn).load()`.

Every selectable field receives an exact-match method:

```rust
Sample::select(conn).is_reference(true);
```

Every selectable field also receives an `_in` method for matching any value from a typed
iterator. Empty iterators match no rows:

```rust
Sample::select(conn).name_in(["sample-a", "sample-b"]);
```

`String` fields also receive a case-insensitive `_contains` method:

```rust
Sample::select(conn).name_contains("reference");
```

Use `_case_insensitive` when the complete string must match while ignoring case:

```rust
Sample::select(conn).name_case_insensitive("REFERENCE");
```

Exact, multi-value, contains, and case-insensitive values are bound SQL parameters. In
particular, `%` and `_` in user input are ordinary characters rather than `LIKE` wildcards.

`Option<T>` fields receive an `_is_null` method in addition to an exact-match method that accepts
`T`:

```rust
let children = BlockGroup::select(conn)
    .parent_block_group_id(parent_id)
    .load()
    .expect("should load child block groups");

let roots = BlockGroup::select(conn)
    .parent_block_group_id_is_null()
    .load()
    .expect("should load root block groups");
```

## Loading one model or all models

Use `get` when a selector must match at most one model. It returns
`Result<Option<Model>, ModelSelectError>`: `None` means no row matched, while multiple matches
produce `ModelSelectError::MultipleResults`.

```rust
let sample = Sample::select(conn)
    .name("sample-a")
    .get()
    .expect("should query sample");
```

The derive infers a field named `id` as the model primary key. Mark a differently named key with
`#[model_select(primary_key)]`; this generates `get_by_id` on the selector:

```rust
let sample = Sample::select(conn)
    .get_by_id("sample-a")
    .expect("should query sample by primary key");
```

Only one field can be marked as the primary key, and skipped fields cannot be primary keys.
Models without an `id` field or an explicit `primary_key` attribute do not receive `get_by_id`.

To load every current model, use the generated convenience method:

```rust
let samples = Sample::all(conn).expect("should load all samples");
```

`Sample::all(conn)` is exactly `Sample::select(conn).load()`. For a historical read, construct the
selector explicitly and call `.with_ref(history_ref).load()`.

## Selecting specific fields

Call `only` after configuring the query to return selected fields instead of complete models. A
single field returns that field's Rust type:

```rust
let names: Vec<String> = Sample::select(conn)
    .name_contains("foo")
    .order_by(SampleSelect::Name, Direction::Asc)
    .only(SampleSelect::Name)
    .load()
    .expect("should load sample names");
```

Pass a tuple of field constants to return typed tuples:

```rust
let samples: Vec<(String, bool)> = Sample::select(conn)
    .order_by(SampleSelect::Name, Direction::Asc)
    .only((SampleSelect::Name, SampleSelect::IsReference))
    .load()
    .expect("should load selected sample fields");
```

Fields may come from both the base selector and joined selectors:

```rust
let rows: Vec<(String, String)> = Sample::select(conn)
    .join_filtered_on(
        SampleSelect::Name,
        BlockGroupSelect::SampleName,
        BlockGroup::select(conn).collection_name("example"),
    )
    .only((SampleSelect::Name, BlockGroupSelect::Name))
    .load()
    .expect("should load fields from both joined models");
```

The field constants determine the return type and preserve the order of values in each tuple. Use
a one-element tuple such as `(SampleSelect::Name,)` when a `Vec<(String,)>` is preferable to a
`Vec<String>`.

`only` is the terminal query-shaping step: apply filters, joins, ordering, limits, and offsets
before it, then call `load`. A projection can contain up to 16 fields. Every projected field's
model must be the base selector or one of its joins.

## Selecting complete joined models

Call `models` with a tuple of model types to return complete models from both sides of a join:

```rust
let rows: Vec<(Sample, BlockGroup)> = Sample::select(conn)
    .join_filtered_on(
        SampleSelect::Name,
        BlockGroupSelect::SampleName,
        BlockGroup::select(conn).collection_name("example"),
    )
    .models::<(Sample, BlockGroup)>()
    .load()
    .expect("should load both joined models");
```

The tuple order determines both the SQL column order and the returned tuple order. Every model must
be the base selector or one of its joins. Like field projections, model tuples support one through
16 entries. Write a one-element tuple such as `.models::<(Sample,)>()` when needed.

The derive generates an offset-aware, fallible row decoder for this API. A model containing a
`#[model_select(skip)]` field does not receive that decoder because a complete value cannot be
constructed from the selected columns.

## Ordering and pagination

Ordering uses generated field constants, so misspelled or removed column names fail at compile
time:

```rust
let samples = Sample::select(conn)
    .order_by(SampleSelect::Name, Direction::CaseInsensitiveAsc)
    .order_by(SampleSelect::IsReference, Direction::Desc)
    .limit(25)
    .offset(50)
    .load()
    .expect("should load ordered samples");
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
    .load()
    .expect("should load historical samples");
```

For the default source, a historical query reads `dolt_at_<table>(:history_ref)`. When selectors
are joined, the same history ref is applied to every source. If both selectors specify refs, they
must be equal.

Models that set `Query::HISTORY_TABLE_NAME` to `None` cannot use `with_ref`.

## Joins

Selectors join on two generated field constants, so the relationship is checked by Rust and does
not require a foreign-key metadata query:

```rust
let samples = Sample::select(conn)
    .name_contains("sample")
    .join_on(SampleSelect::Name, BlockGroupSelect::SampleName)
    .order_by(SampleSelect::Name, Direction::Asc)
    .load()
    .expect("should load joined samples");
```

`join_on` constructs an unfiltered selector for the model carried by its right-hand field. Use
`join_filtered_on` to carry another selector's filters and ordering into the query:

```rust
let samples = Sample::select(conn)
    .join_filtered_on(
        SampleSelect::Name,
        BlockGroupSelect::SampleName,
        BlockGroup::select(conn).collection_name("example"),
    )
    .load()
    .expect("should load samples in the collection");
```

Calling `load` directly on either joined selector returns instances of the selector on the left,
so these queries return `Vec<Sample>`. Use `only` for typed fields from either side of the join, or
`models` for complete joined models.

The join fields are rendered as one quoted equality, such as
`"samples"."name" = "block_groups"."sample_name"`. Repeating an identical join reuses its
insertion-ordered entry and combines the filters. Reusing one alias with a different source or
condition is rejected.

A helper can hide a common relationship while retaining model-specific filter syntax:

```rust
let paths = Path::select(conn)
    .collection_name("example")
    .sample_name("reference")
    .load()
    .expect("should load joined paths");
```

Joined selectors must:

- Use the same database connection.
- Use the same historical ref, when explicitly set on both selectors.
- Use a consistent source and condition when reusing a source alias.
- Leave `limit` and `offset` unset; apply pagination to the outer selector after the join.

Violating these requirements, or joining from a field whose source has not been selected, currently
panics with a descriptive message.

## Error handling

`load()` and `get()` return `ModelSelectError` for database preparation, query execution,
result-row iteration, and projections that refer to a model that was not selected or joined.
`get()` additionally returns `ModelSelectError::MultipleResults` when the selector matches more
than one model. Callers can propagate these errors directly:

```rust,ignore
use gen_models::ModelSelectError;

fn matching_samples(conn: &Connection) -> Result<Vec<Sample>, ModelSelectError> {
    Sample::select(conn).name_contains("foo").load()
}
```

Selector construction errors, such as joining from an unselected model or reusing an alias with a
different condition, are currently programming errors enforced with assertions before `load()`
runs.

## Configuration attributes

Fields use their Rust field name as the SQL column name by default. Override or omit a generated
field with `model_select` attributes:

```rust
#[derive(ModelSelect)]
pub struct Example {
    #[model_select(primary_key)]
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

- `alias = "..."` sets the SQL alias used to qualify generated columns. Aliases and field-level
  `column` values are single SQL identifiers; generated SQL always double-quotes them and treats
  punctuation such as `.` as part of the identifier.
- `source = path::to::function` supplies a `fn(Option<&str>) -> String` that renders the `FROM`
  source. Its SQL must expose the configured alias and honor the optional history ref when needed.
- `select = "..."` replaces the default `<alias>.*` select list. Its column order must still match
  `Query::process_row`.

`source` and `select` are trusted, compile-time raw SQL escape hatches rather than identifiers, as
are clauses passed directly to `SqlFilter::new`. Their authors are responsible for quoting every
identifier and binding runtime values. They must never be populated from runtime input.

## How it works

The implementation is split between compile-time generation and runtime execution.

At compile time, `ModelSelect`:

1. Accepts a non-generic struct with named fields.
2. Reads the model and field-level `model_select` attributes.
3. Generates the selector struct, typed field constants, filter methods, and `SelectQuery`
   implementation.
4. Generates `Model::select(conn)`, `Model::all(conn)`, selector `get()`, and selector
   `get_by_id(...)` when a primary key is available.

At runtime, [`gen-models::select`](../gen-models/src/select.rs):

1. Stores generated filters, ordering, joins, pagination, and the optional history ref.
2. Builds requested join conditions from typed field constants supplied by the caller.
3. Quotes every structured table, alias, and column identifier, then renders one `SELECT` statement
   for the base model and its joined sources.
4. Binds every runtime value through `rusqlite`.
5. Maps full base-model loads through `Query::process_row`, or typed field and model projections
   through fallible decoders generated for their selected types.

The runtime support cannot live in this proc-macro crate. A proc-macro executes inside the compiler
and can export procedural macros, but it cannot export the normal reusable runtime types needed by
all generated selectors. It also has no access to the query's live database connection. Keeping
the SQL renderer in ordinary Rust provides one implementation that can be tested and reviewed
without expanding every derive.

## Current limitations

- Only structs with named fields are supported.
- Generic model structs are not supported.
- At least one field must remain selectable after applying `skip`.
- Each `join_on` call expresses one equality; composite join conditions are not yet supported.
- A source alias can only refer to one source and join condition in a query.
- `only` supports projections of one through 16 fields from selected and joined models.
- `models` supports tuples of one through 16 selected and joined models without skipped fields.
- `Query::process_row` returns a model directly, so a model implementation that panics while
  decoding a row cannot be converted into `ModelSelectError` without changing the `Query` trait.
- Generated code references `gen_models`, so downstream consumers should use the macro through the
  `gen-models` crate under its standard crate name.
